#!/usr/bin/env python3
"""Estimate and center chess phase Elo with a fixed candidate-game reference."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import chess

from analyze import PHASES, game_to_sgf, phase_boundaries, player_color, read_games
from overall import BUCKET_PATTERN, estimate_elo


MOVE_PATTERN = re.compile(r";[BW]\[([a-h][1-8][a-h][1-8][qrbn]?)\]")


def candidate_boards(sgf: str) -> list[chess.Board]:
  board = chess.Board()
  boards = []
  for uci in MOVE_PATTERN.findall(sgf):
    board.push(board.parse_uci(uci))
    boards.append(board.copy(stack=False))
  if len(boards) < 2:
    raise ValueError("candidate game needs at least two plies")
  return boards


def phase_totals(scorer, sgf: str, boards: list[chess.Board]) -> dict[str, dict[str, tuple[float, int]]]:
  from prediction_testing.sgf import game_prefix

  middle, end = phase_boundaries(boards)
  stops = sorted({0, len(boards), *(index for index in (middle, end) if index is not None)})
  cumulative = {"white": (0.0, 0), "black": (0.0, 0)}
  totals = {color: {phase: (0.0, 0) for phase in PHASES}
            for color in ("white", "black")}
  for start, stop in zip(stops, stops[1:]):
    observed = max(2, stop)
    averages = scorer.score_sgf(game_prefix(sgf, observed))
    phase = "endgame" if end is not None and start >= end else (
      "midgame" if middle is not None and start >= middle else "opening")
    for color in ("white", "black"):
      count = (stop + (color == "white")) // 2
      value = float(averages[color]) * count
      old_value, old_count = cumulative[color]
      if not math.isfinite(value):
        raise ValueError("model returned non-finite strength")
      totals[color][phase] = (value - old_value, count - old_count)
      cumulative[color] = (value, count)
  return totals


def add_totals(target: dict[str, list[float]], source: dict[str, tuple[float, int]]) -> None:
  for phase in PHASES:
    target[phase][0] += source[phase][0]
    target[phase][1] += source[phase][1]


def empty_totals() -> dict[str, list[float]]:
  return {phase: [0.0, 0] for phase in PHASES}


def phase_ratings(totals: dict[str, list[float]], calibration: list[dict]) -> dict[str, float | None]:
  return {phase: float(estimate_elo(total / count, calibration)["estimated_elo"])
          if count else None for phase, (total, count) in totals.items()}


def center_ratings(ratings: dict[str, float | None], reference_data: dict) -> dict[str, float | None]:
  return {phase: (rating - reference_data["phase_elo"][phase] + reference_data["mean_elo"])
          if rating is not None else None for phase, rating in ratings.items()}


def reference(scorer, candidate_dir: Path, calibration: list[dict]) -> dict:
  totals = empty_totals()
  games = skipped = 0
  for path in sorted(path for path in candidate_dir.glob("*.txt")
                     if BUCKET_PATTERN.match(path.stem)):
    with path.open(encoding="utf-8", errors="replace") as stream:
      for line_number, sgf in enumerate(stream, start=1):
        if not sgf.strip():
          continue
        games += 1
        try:
          scores = phase_totals(scorer, sgf, candidate_boards(sgf))
          for color in ("white", "black"):
            add_totals(totals, scores[color])
        except (ValueError, RuntimeError) as error:
          skipped += 1
          print(f"Skip reference {path}:{line_number}: {error}", file=sys.stderr)
  if any(not totals[phase][1] for phase in PHASES):
    raise ValueError("reference lacks one or more phases")
  ratings = phase_ratings(totals, calibration)
  return {"phase_elo": ratings, "phase_moves": {phase: int(totals[phase][1]) for phase in PHASES},
          "games": games - skipped, "skipped_games": skipped,
          "mean_elo": sum(ratings.values()) / 3}


def player_report(scorer, pgn: Path, player: str, calibration: list[dict], reference_data: dict) -> dict:
  totals = empty_totals()
  matched = skipped = 0
  paths = [pgn] if pgn.is_file() else sorted(pgn.rglob("*.pgn"))
  for path in paths:
    for game_index, game in enumerate(read_games(path), start=1):
      color = player_color(game, player)
      if color is None:
        continue
      matched += 1
      try:
        sgf, positions = game_to_sgf(game)
        boards = []
        board = game.board()
        for move in game.mainline_moves():
          board.push(move)
          boards.append(board.copy(stack=False))
        if len(positions) < 2:
          raise ValueError("game needs at least two plies")
        add_totals(totals, phase_totals(scorer, sgf, boards)[color])
      except (ValueError, RuntimeError) as error:
        skipped += 1
        print(f"Skip player {path}:{game_index}: {error}", file=sys.stderr)
  ratings = phase_ratings(totals, calibration)
  normalized = center_ratings(ratings, reference_data)
  complete = all(rating is not None for rating in ratings.values())
  return {"player": player, "matched_games": matched, "analyzed_games": matched - skipped,
          "skipped_games": skipped, "phase_elo": ratings,
          "phase_norm_elo": normalized,
          "phase_moves": {phase: int(totals[phase][1]) for phase in PHASES},
          "total_elo": sum(ratings.values()) / 3 if complete else None}


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("pgn")
  parser.add_argument("player")
  parser.add_argument("--config", required=True)
  parser.add_argument("--checkpoint", required=True)
  parser.add_argument("--calibration", type=Path, required=True)
  parser.add_argument("--candidate-dir", type=Path, required=True)
  parser.add_argument("--reference", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--gpu-id", type=int, default=0)
  args = parser.parse_args()
  sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
  from build.chess import strength_py

  print(f"Config: model={args.checkpoint}, calibration={args.calibration}, "
        f"reference={args.reference}, candidate_dir={args.candidate_dir}, "
        f"player={args.player}, pgn={args.pgn}, gpu_id={args.gpu_id}")
  scorer = strength_py.StrengthScorer(args.config, args.checkpoint, args.gpu_id)
  if strength_py.get_bt_use_weight():
    raise ValueError("phase aggregation requires bt_use_weight=false")
  calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
  if args.reference.exists():
    reference_data = json.loads(args.reference.read_text(encoding="utf-8"))
    expected = {"checkpoint": args.checkpoint, "calibration": str(args.calibration),
                "candidate_dir": str(args.candidate_dir), "divider": "lichess",
                "elo_conversion": "linear_extrapolation"}
    if any(reference_data.get(key) != value for key, value in expected.items()):
      raise ValueError("reference configuration does not match this run")
  else:
    reference_data = reference(scorer, args.candidate_dir, calibration)
    reference_data.update({"checkpoint": args.checkpoint, "calibration": str(args.calibration),
                           "candidate_dir": str(args.candidate_dir), "divider": "lichess",
                           "elo_conversion": "linear_extrapolation"})
    args.reference.parent.mkdir(parents=True, exist_ok=True)
    args.reference.write_text(json.dumps(reference_data, indent=2), encoding="utf-8")
  print(f"Reference: games={reference_data['games']}, phase_moves={reference_data['phase_moves']}, "
        f"phase_elo={reference_data['phase_elo']}, mean_elo={reference_data['mean_elo']:.1f}")
  result = player_report(scorer, Path(args.pgn), args.player, calibration, reference_data)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  with args.output.open("w", newline="", encoding="utf-8") as stream:
    row = {key: value for key, value in result.items() if not isinstance(value, dict)}
    for phase in PHASES:
      row[f"{phase}_elo"] = result["phase_elo"][phase]
      row[f"{phase}_norm_elo"] = result["phase_norm_elo"][phase]
      row[f"{phase}_moves"] = result["phase_moves"][phase]
    writer = csv.DictWriter(stream, fieldnames=row.keys())
    writer.writeheader()
    writer.writerow(row)
  print(f"Done: games={result['analyzed_games']}, phase_moves={result['phase_moves']}, "
        f"phase_elo={result['phase_elo']}, normalized={result['phase_norm_elo']}, "
        f"total_elo={result['total_elo']}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
