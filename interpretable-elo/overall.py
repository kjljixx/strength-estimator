#!/usr/bin/env python3
"""Estimate one player's overall strength from all moves in their PGNs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze import game_to_sgf, player_color, read_games


BUCKET_PATTERN = re.compile(r"^(\d+)_(\d+)$")


def color_move_count(total_plies: int, color: str) -> int:
  return (total_plies + 1) // 2 if color == "white" else total_plies // 2


def weighted_mean(weighted_scores: list[tuple[float, int]]) -> float:
  total_weight = sum(weight for _, weight in weighted_scores)
  if total_weight == 0:
    raise ValueError("no moves available to score")
  return sum(score * weight for score, weight in weighted_scores) / total_weight


def score_player(scorer, pgn_dir: Path, player: str) -> dict[str, float | int]:
  weighted_scores: list[tuple[float, int]] = []
  recorded_ratings: list[int] = []
  matched_games = skipped_games = 0
  if pgn_dir.is_file() and pgn_dir.suffix.lower() == ".txt":
    return score_sgf_player(scorer, pgn_dir, player)
  paths = [pgn_dir] if pgn_dir.is_file() else sorted(pgn_dir.rglob("*.pgn"))
  for path in paths:
    for game_index, game in enumerate(read_games(path), start=1):
      color = player_color(game, player)
      if color is None:
        continue
      matched_games += 1
      rating = game.headers.get("WhiteElo" if color == "white" else "BlackElo", "")
      if rating.isdigit():
        recorded_ratings.append(int(rating))
      try:
        sgf, positions = game_to_sgf(game)
        move_count = color_move_count(len(positions), color)
        if move_count == 0 or len(positions) < 2:
          raise ValueError("game needs at least two plies")
        score = float(scorer.score_sgf(sgf)[color])
        if not math.isfinite(score):
          raise ValueError("model returned non-finite strength")
        weighted_scores.append((score, move_count))
      except (ValueError, RuntimeError) as error:
        skipped_games += 1
        print(f"Skip {path}:{game_index}: {error}", file=sys.stderr)
  return {
    "matched_games": matched_games,
    "analyzed_games": len(weighted_scores),
    "skipped_games": skipped_games,
    "moves": sum(weight for _, weight in weighted_scores),
    "mean_score": weighted_mean(weighted_scores),
    "median_recorded_rating": statistics.median(recorded_ratings) if recorded_ratings else "",
  }


def score_sgf_player(scorer, path: Path, player: str) -> dict[str, float | int]:
  from prediction_testing.sgf import move_count

  weighted_scores: list[tuple[float, int]] = []
  recorded_ratings: list[int] = []
  matched_games = skipped_games = 0
  wanted = player.casefold()
  with path.open(encoding="utf-8", errors="replace") as stream:
    for line_number, sgf in enumerate(stream, start=1):
      white = re.search(r"PW\[([^]]*)\]", sgf)
      black = re.search(r"PB\[([^]]*)\]", sgf)
      if white and white.group(1).casefold() == wanted:
        color, rating_tag = "white", "WR"
      elif black and black.group(1).casefold() == wanted:
        color, rating_tag = "black", "BR"
      else:
        continue
      matched_games += 1
      rating = re.search(rf"{rating_tag}\[(\d+)\]", sgf)
      if rating:
        recorded_ratings.append(int(rating.group(1)))
      try:
        plies = move_count(sgf)
        moves = color_move_count(plies, color)
        if moves == 0 or plies < 2:
          raise ValueError("game needs at least two plies")
        score = float(scorer.score_sgf(sgf)[color])
        if not math.isfinite(score):
          raise ValueError("model returned non-finite strength")
        weighted_scores.append((score, moves))
      except (ValueError, RuntimeError) as error:
        skipped_games += 1
        print(f"Skip {path}:{line_number}: {error}", file=sys.stderr)
  return {
    "matched_games": matched_games,
    "analyzed_games": len(weighted_scores),
    "skipped_games": skipped_games,
    "moves": sum(weight for _, weight in weighted_scores),
    "mean_score": weighted_mean(weighted_scores),
    "median_recorded_rating": statistics.median(recorded_ratings) if recorded_ratings else "",
  }


def score_candidate_file(scorer, path: Path) -> dict[str, float | int]:
  match = BUCKET_PATTERN.match(path.stem)
  if not match:
    raise ValueError(f"candidate filename must be LOW_HIGH.txt: {path.name}")
  low, high = map(int, match.groups())
  weighted_scores: list[tuple[float, int]] = []
  games = skipped = 0
  with path.open(encoding="utf-8", errors="replace") as stream:
    for line_number, sgf in enumerate(stream, start=1):
      sgf = sgf.strip()
      if not sgf:
        continue
      games += 1
      try:
        from prediction_testing.sgf import move_count

        plies = move_count(sgf)
        if plies < 2:
          raise ValueError("game needs at least two plies")
        scores = scorer.score_sgf(sgf)
        weighted_scores.extend((
          (float(scores["white"]), (plies + 1) // 2),
          (float(scores["black"]), plies // 2),
        ))
      except (ValueError, RuntimeError) as error:
        skipped += 1
        print(f"Skip {path}:{line_number}: {error}", file=sys.stderr)
  return {
    "low": low,
    "high": high,
    "center": (low + high) / 2,
    "games": games - skipped,
    "moves": sum(weight for _, weight in weighted_scores),
    "mean_score": weighted_mean(weighted_scores),
  }


def build_calibration(scorer, candidate_dir: Path) -> list[dict[str, float | int]]:
  paths = sorted(
    (path for path in candidate_dir.glob("*.txt") if BUCKET_PATTERN.match(path.stem)),
    key=lambda path: int(path.stem.split("_", 1)[0]),
  )
  if not paths:
    raise ValueError(f"no LOW_HIGH.txt candidate files in {candidate_dir}")
  calibration = []
  for path in paths:
    print(f"Calibrating {path.name}")
    calibration.append(score_candidate_file(scorer, path))
  return calibration


def estimate_elo(score: float, calibration: list[dict[str, float | int]]) -> dict[str, object]:
  nearest = min(calibration, key=lambda row: abs(score - float(row["mean_score"])))
  ordered = sorted(calibration, key=lambda row: float(row["mean_score"]))
  if score <= float(ordered[0]["mean_score"]):
    estimated = float(ordered[0]["center"])
    status = "below_range"
  elif score >= float(ordered[-1]["mean_score"]):
    estimated = float(ordered[-1]["center"])
    status = "above_range"
  else:
    status = "interpolated"
    for lower, upper in zip(ordered, ordered[1:]):
      lower_score = float(lower["mean_score"])
      upper_score = float(upper["mean_score"])
      if lower_score <= score <= upper_score:
        fraction = (score - lower_score) / (upper_score - lower_score)
        estimated = float(lower["center"]) + fraction * (
          float(upper["center"]) - float(lower["center"])
        )
        break
  return {
    "estimated_elo": estimated,
    "estimate_status": status,
    "nearest_bucket": f"{nearest['low']}-{nearest['high']}",
  }


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("pgn_dir", type=Path)
  parser.add_argument("player")
  parser.add_argument("--config", type=Path, required=True)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--candidate-dir", type=Path)
  parser.add_argument("--calibration", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--gpu-id", type=int, default=0)
  return parser


def main() -> int:
  args = build_parser().parse_args()
  root = Path(__file__).resolve().parents[1]
  sys.path.insert(0, str(root))
  from build.chess import strength_py

  print(
    f"Config: player={args.player!r}, pgn_dir={args.pgn_dir}, gpu_id={args.gpu_id}, "
    f"calibration={args.calibration}"
  )
  scorer = strength_py.StrengthScorer(str(args.config), str(args.checkpoint), args.gpu_id)
  if strength_py.get_bt_use_weight():
    raise ValueError("standard-compatible aggregation requires bt_use_weight=false")

  if args.calibration.exists():
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    print(f"Loaded {len(calibration)} calibration buckets")
  else:
    if args.candidate_dir is None:
      raise ValueError("--candidate-dir is required when calibration does not exist")
    calibration = build_calibration(scorer, args.candidate_dir)
    args.calibration.parent.mkdir(parents=True, exist_ok=True)
    args.calibration.write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    print(f"Wrote {len(calibration)} calibration buckets to {args.calibration}")

  result = {"player": args.player, **score_player(scorer, args.pgn_dir, args.player)}
  result.update(estimate_elo(float(result["mean_score"]), calibration))
  args.output.parent.mkdir(parents=True, exist_ok=True)
  with args.output.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=result.keys())
    writer.writeheader()
    writer.writerow(result)
  print(
    f"Done: games={result['analyzed_games']}, moves={result['moves']}, "
    f"score={result['mean_score']:.6f}, estimated_elo={result['estimated_elo']:.1f}, "
    f"nearest_bucket={result['nearest_bucket']}, status={result['estimate_status']}"
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
