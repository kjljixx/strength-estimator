#!/usr/bin/env python3
"""Plot directly calibrated, reference-centered per-move Elo distributions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from analyze import PHASES
from overall import estimate_elo


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("scores_dir", type=Path)
  parser.add_argument("--reference", type=Path, required=True)
  parser.add_argument("--calibration", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  args = parser.parse_args()
  reference = json.loads(args.reference.read_text(encoding="utf-8"))
  calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
  shifts = {phase: reference["mean_elo"] - reference["phase_elo"][phase]
            for phase in PHASES}
  print(f"Config: scores_dir={args.scores_dir}, reference={args.reference}, "
        f"calibration={args.calibration}, output_dir={args.output_dir}, "
        f"phase_shifts={shifts}")
  args.output_dir.mkdir(parents=True, exist_ok=True)
  sns.set_theme(style="whitegrid")
  colors = {"opening": "#31708e", "midgame": "#c17b24", "endgame": "#53804b"}
  players = {}
  for path in sorted(args.scores_dir.glob("*.csv")):
    moves = pd.read_csv(path)
    if moves.empty:
      raise ValueError(f"no move scores in {path}")
    moves["move_elo"] = moves["strength"].map(
      lambda score: estimate_elo(float(score), calibration)["estimated_elo"])
    moves["normalized_move_elo"] = moves["move_elo"] + moves["phase"].map(shifts)
    if moves["normalized_move_elo"].isna().any():
      raise ValueError(f"unknown phase in {path}")
    moves.to_csv(args.output_dir / f"{path.stem}-moves.csv", index=False)
    players[path.stem] = moves

  if not players:
    raise ValueError(f"no move-score CSV files in {args.scores_dir}")
  all_values = pd.concat([moves["normalized_move_elo"] for moves in players.values()])
  lower, upper = all_values.quantile([0.01, 0.99])
  for player, moves in players.items():
    visible = moves[moves["normalized_move_elo"].between(lower, upper)]
    figure, axis = plt.subplots(figsize=(9, 5))
    sns.histplot(data=visible, x="normalized_move_elo", hue="phase",
                 hue_order=PHASES, palette=colors, bins=80, stat="density",
                 common_norm=False, element="step", fill=False, linewidth=1.7,
                 ax=axis)
    axis.set(title=f"{player}: normalized Elo per move",
             xlabel="Reference-centered Elo equivalent per move",
             ylabel="Density within each phase")
    axis.set_xlim(lower, upper)
    counts = moves["phase"].value_counts()
    figure.text(0.5, 0.01,
                f"Common central 98% range: {lower:,.0f} to {upper:,.0f}; tails omitted. "
                f"Moves: O {counts.get('opening', 0):,}, M {counts.get('midgame', 0):,}, "
                f"E {counts.get('endgame', 0):,}. Direct per-move conversion.",
                ha="center", fontsize=9)
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    figure.savefig(args.output_dir / f"{player}.png", dpi=200)
    plt.close(figure)
    print(f"Done: player={player}, moves={len(moves)}, "
          f"phase_moves={moves['phase'].value_counts().to_dict()}, "
          f"plot_range=({lower:.1f}, {upper:.1f})")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
