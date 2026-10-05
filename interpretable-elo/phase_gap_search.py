#!/usr/bin/env python3
"""Rank players by midgame minus endgame normalized Elo, using a fixed phase reference."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from analyze import PHASES
from phase_elo import player_report

FIELDS = ["player", "site_rapid_rating", "analyzed_games", "total_elo", "opening_norm_elo",
          "midgame_norm_elo", "endgame_norm_elo", "midgame_minus_endgame", "midgame_moves", "endgame_moves"]


def site_rating(pgn: Path, player: str) -> float | None:
  ratings = []
  lines = pgn.read_text(encoding="utf-8", errors="replace").splitlines()
  white = black = None
  for line in lines:
    if line.startswith('[White "'):
      white = line.split('"')[1]
    elif line.startswith('[Black "'):
      black = line.split('"')[1]
    elif line.startswith('[WhiteElo "') and white == player:
      ratings.append(int(line.split('"')[1]))
    elif line.startswith('[BlackElo "') and black == player:
      ratings.append(int(line.split('"')[1]))
  return sum(ratings[:20]) / len(ratings[:20]) if ratings else None


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("pgn_dir", type=Path)
  parser.add_argument("--config", required=True)
  parser.add_argument("--checkpoint", required=True)
  parser.add_argument("--calibration", type=Path, required=True)
  parser.add_argument("--reference", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--gpu-id", type=int, default=0)
  args = parser.parse_args()
  sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
  from build.chess import strength_py

  files = sorted(args.pgn_dir.glob("*-rapid-*.pgn"))
  print(f"Config: model={args.checkpoint}, calibration={args.calibration}, reference={args.reference}, "
        f"pgn_dir={args.pgn_dir}, players={len(files)}, gpu_id={args.gpu_id}")
  scorer = strength_py.StrengthScorer(args.config, args.checkpoint, args.gpu_id)
  if strength_py.get_bt_use_weight():
    raise ValueError("phase aggregation requires bt_use_weight=false")
  calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
  reference_data = json.loads(args.reference.read_text(encoding="utf-8"))
  if reference_data["checkpoint"] != args.checkpoint:
    raise ValueError("reference checkpoint does not match this run")

  rows = []
  args.output.parent.mkdir(parents=True, exist_ok=True)
  for index, path in enumerate(files, start=1):
    player = path.name.split("-rapid-")[0]
    result = player_report(scorer, path, player, calibration, reference_data)
    normalized = result["phase_norm_elo"]
    if any(normalized[phase] is None for phase in PHASES):
      print(f"[{index}/{len(files)}] {player}: skipped, missing phase, moves={result['phase_moves']}")
      continue
    rows.append({"player": player, "site_rapid_rating": site_rating(path, player),
                 "analyzed_games": result["analyzed_games"], "total_elo": result["total_elo"],
                 "opening_norm_elo": normalized["opening"], "midgame_norm_elo": normalized["midgame"],
                 "endgame_norm_elo": normalized["endgame"],
                 "midgame_minus_endgame": normalized["midgame"] - normalized["endgame"],
                 "midgame_moves": result["phase_moves"]["midgame"],
                 "endgame_moves": result["phase_moves"]["endgame"]})
    print(f"[{index}/{len(files)}] {player}: site={rows[-1]['site_rapid_rating']}, "
          f"mid-end={rows[-1]['midgame_minus_endgame']:.0f}, games={result['analyzed_games']}")
    rows.sort(key=lambda row: row["midgame_minus_endgame"], reverse=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
      writer = csv.DictWriter(stream, fieldnames=FIELDS)
      writer.writeheader()
      writer.writerows(rows)
  print(f"Done: {len(rows)}/{len(files)} players ranked; top={[row['player'] for row in rows[:5]]}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
