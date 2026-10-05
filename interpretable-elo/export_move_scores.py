#!/usr/bin/env python3
"""Export one player's per-move strength scores and Lichess phases."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

from analyze import PHASES, game_to_sgf, player_color, read_games, score_game


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("pgn", type=Path)
  parser.add_argument("player")
  parser.add_argument("--config", required=True)
  parser.add_argument("--checkpoint", required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--gpu-id", type=int, default=0)
  args = parser.parse_args()
  sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
  from build.chess import strength_py

  print(f"Config: pgn={args.pgn}, player={args.player}, model={args.checkpoint}, "
        f"output={args.output}, gpu_id={args.gpu_id}", flush=True)
  scorer = strength_py.StrengthScorer(args.config, args.checkpoint, args.gpu_id)
  if strength_py.get_bt_use_weight():
    raise ValueError("per-move scoring requires bt_use_weight=false")

  paths = [args.pgn] if args.pgn.is_file() else sorted(args.pgn.rglob("*.pgn"))
  args.output.parent.mkdir(parents=True, exist_ok=True)
  matched = skipped = 0
  counts: Counter[str] = Counter()
  with args.output.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=("game", "ply", "phase", "strength"))
    writer.writeheader()
    for path in paths:
      for game_index, game in enumerate(read_games(path), start=1):
        color = player_color(game, args.player)
        if color is None:
          continue
        matched += 1
        try:
          sgf, positions = game_to_sgf(game)
          rows = [row for row in score_game(scorer, sgf, positions)
                  if row["color"] == color]
          writer.writerows({"game": game_index, "ply": row["ply"],
                            "phase": row["phase"], "strength": row["strength"]}
                           for row in rows)
          counts.update(row["phase"] for row in rows)
        except (ValueError, RuntimeError) as error:
          skipped += 1
          print(f"Skip {path}:{game_index}: {error}", file=sys.stderr, flush=True)
        if matched % 100 == 0:
          print(f"Progress: matched_games={matched}, scored_moves={sum(counts.values())}",
                flush=True)
  print(f"Done: matched_games={matched}, analyzed_games={matched - skipped}, "
        f"skipped_games={skipped}, phase_moves="
        f"{dict((phase, counts[phase]) for phase in PHASES)}, output={args.output}",
        flush=True)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
