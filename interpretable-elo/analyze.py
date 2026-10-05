#!/usr/bin/env python3
"""Map a player's chess positions to move-strength scores."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import chess
import chess.pgn

from phase_rules import phase_boundaries, phase_for, PHASES


def instantaneous_score(average: float, previous_average: float, move_count: int) -> float:
  return move_count * average - (move_count - 1) * previous_average


def engine_move_uci(board: chess.Board, move: chess.Move) -> str:
  if not board.is_castling(move):
    return move.uci()
  rook_square = chess.H1 if board.is_kingside_castling(move) else chess.A1
  if board.turn == chess.BLACK:
    rook_square += 56
  return chess.square_name(move.from_square) + chess.square_name(rook_square)


def game_to_sgf(game: chess.pgn.Game) -> tuple[str, list[dict[str, object]]]:
  board = game.board()
  if board.fen() != chess.Board().fen():
    raise ValueError("non-standard starting position")

  moves: list[str] = []
  positions: list[dict[str, object]] = []
  boards: list[chess.Board] = []
  for ply, move in enumerate(game.mainline_moves(), start=1):
    positions.append({
      "ply": ply,
      "fullmove": board.fullmove_number,
      "color": "white" if board.turn == chess.WHITE else "black",
      "fen_before": board.fen(),
      "move_uci": move.uci(),
    })
    moves.append(f";{'B' if board.turn == chess.WHITE else 'W'}[{engine_move_uci(board, move)}]")
    board.push(move)
    boards.append(board.copy(stack=False))

  middle, end = phase_boundaries(boards)
  for index, position in enumerate(positions):
    position["phase"] = phase_for(index, middle, end)

  result = {"1-0": "1.0", "0-1": "-1.0"}.get(game.headers.get("Result"), "0.0")
  root = (
    f";GM[chess]RE[{result}]"
    f"PW[{game.headers.get('White', '')}]PB[{game.headers.get('Black', '')}]"
  )
  return f"({root}{''.join(moves)})", positions


def player_color(game: chess.pgn.Game, player: str) -> str | None:
  wanted = player.casefold()
  if game.headers.get("White", "").casefold() == wanted:
    return "white"
  if game.headers.get("Black", "").casefold() == wanted:
    return "black"
  return None


def score_game(scorer, sgf: str, positions: list[dict[str, object]]) -> list[dict[str, object]]:
  if len(positions) < 2:
    raise ValueError("game needs at least two plies")

  from prediction_testing.sgf import game_prefix

  previous_averages = {"white": 0.0, "black": 0.0}
  counts = {"white": 0, "black": 0}
  rows: list[dict[str, object]] = []
  for observed_plies in range(2, len(positions) + 1):
    averages = scorer.score_sgf(game_prefix(sgf, observed_plies))
    new_positions = positions[:2] if observed_plies == 2 else [positions[observed_plies - 1]]
    for position in new_positions:
      color = str(position["color"])
      counts[color] += 1
      score = instantaneous_score(
        averages[color],
        previous_averages[color],
        counts[color],
      )
      if not math.isfinite(score):
        raise ValueError("model returned non-finite strength")
      rows.append({**position, "strength": score})
    for color in ("white", "black"):
      previous_averages[color] = averages[color]
  return rows


def read_games(path: Path):
  with path.open(encoding="utf-8-sig", errors="replace") as stream:
    while game := chess.pgn.read_game(stream):
      yield game


def write_outputs(
  rows: list[dict[str, object]],
  output_dir: Path,
) -> None:
  position_groups: dict[str, list[float]] = defaultdict(list)
  phase_groups: dict[str, list[float]] = defaultdict(list)
  for row in rows:
    phase = str(row["phase"])
    key = f"{row['fen_before']} | {row['move_uci']}"
    position_groups[key].append(float(row["strength"]))
    phase_groups[phase].append(float(row["strength"]))

  position_map = {
    key: {"average_strength": sum(scores) / len(scores), "count": len(scores)}
    for key, scores in position_groups.items()
  }
  output_dir.mkdir(parents=True, exist_ok=True)
  (output_dir / "positions.json").write_text(
    json.dumps(position_map, indent=2, sort_keys=True),
    encoding="utf-8",
  )
  with (output_dir / "phases.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=("phase", "average_strength", "move_count"))
    writer.writeheader()
    for phase in PHASES:
      scores = phase_groups[phase]
      writer.writerow({
        "phase": phase,
        "average_strength": sum(scores) / len(scores) if scores else "",
        "move_count": len(scores),
      })


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("pgn_dir", type=Path)
  parser.add_argument("player")
  parser.add_argument("--config", type=Path, required=True)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, default=Path("interpretable-elo-output"))
  parser.add_argument("--gpu-id", type=int, default=0)
  return parser


def main() -> int:
  args = build_parser().parse_args()
  paths = sorted(args.pgn_dir.rglob("*.pgn"))
  print(
    f"Config: pgn_files={len(paths)}, player={args.player!r}, gpu_id={args.gpu_id}, "
    f"phase_divider=lichess"
  )

  sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
  from build.chess import strength_py

  scorer = strength_py.StrengthScorer(str(args.config), str(args.checkpoint), args.gpu_id)
  if strength_py.get_bt_use_weight():
    raise ValueError("instantaneous scoring requires bt_use_weight=false")
  rows: list[dict[str, object]] = []
  matched = skipped = 0
  for path in paths:
    for game_index, game in enumerate(read_games(path), start=1):
      color = player_color(game, args.player)
      if color is None:
        continue
      matched += 1
      try:
        sgf, positions = game_to_sgf(game)
        rows.extend(row for row in score_game(scorer, sgf, positions) if row["color"] == color)
      except (ValueError, RuntimeError) as error:
        skipped += 1
        print(f"Skip {path}:{game_index}: {error}", file=sys.stderr)

  write_outputs(rows, args.output_dir)
  print(
    f"Done: matched_games={matched}, skipped_games={skipped}, scored_moves={len(rows)}, "
    f"unique_positions={len({(row['fen_before'], row['move_uci']) for row in rows})}, "
    f"output_dir={args.output_dir}"
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
