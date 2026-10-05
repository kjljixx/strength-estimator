#!/usr/bin/env python3
"""Write a Markdown table of raw and normalized phase Elo per player from a ranked CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

PHASES = ("opening", "midgame", "endgame")
HEADERS = ["Player", "Site rapid", "Games", "Total", "Raw open", "Raw mid", "Raw end", "Raw mid−end",
           "Norm open", "Norm mid", "Norm end", "Norm mid−end"]


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("ranked_csv", type=Path)
  parser.add_argument("--reference", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  reference = json.loads(args.reference.read_text(encoding="utf-8"))
  shift = {phase: reference["phase_elo"][phase] - reference["mean_elo"] for phase in PHASES}

  rows = []
  with args.ranked_csv.open(encoding="utf-8") as stream:
    for row in csv.DictReader(stream):
      raw = {phase: float(row[f"{phase}_norm_elo"]) + shift[phase] for phase in PHASES}
      norm = {phase: float(row[f"{phase}_norm_elo"]) for phase in PHASES}
      rows.append((raw["midgame"] - raw["endgame"], row, raw, norm))
  rows.sort(key=lambda item: item[0], reverse=True)

  lines = [f"# Midgame vs endgame Elo, {len(rows)} low-rated Lichess players", "",
           "Sorted by raw midgame − endgame gap, highest first. Site rapid = mean rating over the player's "
           "first 20 games in the file. Raw = model Elo per phase. Norm = raw − reference phase Elo + reference mean "
           f"(reference: opening {reference['phase_elo']['opening']:.0f}, midgame {reference['phase_elo']['midgame']:.0f}, "
           f"endgame {reference['phase_elo']['endgame']:.0f}, mean {reference['mean_elo']:.0f}). "
           "Total = mean of the three raw phase Elos.", "",
           "| " + " | ".join(HEADERS) + " |", "|---|" + "---:|" * (len(HEADERS) - 1)]
  for raw_gap, row, raw, norm in rows:
    cells = [row["player"], f"{float(row['site_rapid_rating']):.0f}", row["analyzed_games"],
             f"{float(row['total_elo']):.0f}", *(f"{raw[phase]:.0f}" for phase in PHASES), f"{raw_gap:+.0f}",
             *(f"{norm[phase]:.0f}" for phase in PHASES), f"{norm['midgame'] - norm['endgame']:+.0f}"]
    lines.append("| " + " | ".join(cells) + " |")
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
  print(f"Wrote {len(rows)} players to {args.output}; top raw gap: {rows[0][1]['player']} {rows[0][0]:+.0f}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
