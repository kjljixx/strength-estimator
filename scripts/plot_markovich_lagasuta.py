#!/usr/bin/env python3
"""Plot the supplied game's strength trajectory against its engine evaluation."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ENGINE_EVALS = [
  0.18, 0.32, 0.20, 0.31, 0.22, 0.19, 0.18, 0.49, 0.50, 0.47,
  0.69, 0.55, 0.52, 0.59, 0.65, 0.68, 0.59, 0.87, 0.91, 2.03,
  2.19, 2.03, 1.80, 2.22, 2.10, 2.00, 2.06, 5.47, 5.50, 5.60,
  6.09, 5.96, 5.47, 6.40, 3.07, 2.97, 2.37, 2.92, 2.28, 2.17,
  2.29, 3.56, 3.20, 5.85, 5.63, 5.86, 5.19, 5.66, 5.77, 5.77,
  5.98, 7.05, 6.96, 7.41, 7.15, 6.71, 6.41, 6.69, 6.32, 8.0,
  7.0, 7.0, 6.0,
]


def main() -> None:
  root = Path(__file__).resolve().parents[1]
  csv_path = root / "markovich0104_vs_lagasuta_progressive.csv"
  output_path = root / "markovich0104_vs_lagasuta_progressive.png"
  data = pd.read_csv(csv_path)
  data["engine_eval"] = ENGINE_EVALS[1:]
  data["engine_eval_clipped"] = data["engine_eval"].clip(-5, 5)

  plt.style.use("seaborn-v0_8-whitegrid")
  figure, strength_axis = plt.subplots(figsize=(11, 6.4))
  engine_axis = strength_axis.twinx()
  strength_line = strength_axis.plot(
    data["observed_full_moves"],
    data["white_instantaneous_elo_difference"],
    color="#1478c8",
    linewidth=1.7,
    label="Instantaneous winner Elo difference",
  )[0]
  engine_line = engine_axis.plot(
    data["observed_full_moves"],
    data["engine_eval_clipped"],
    color="#1478c8",
    linestyle="--",
    linewidth=1.7,
    label="Engine eval (winner perspective, clipped ±5)",
  )[0]
  strength_axis.axhline(0, color="#1478c8", linestyle=":", linewidth=1)
  strength_axis.set(
    xlabel="Full moves observed",
    ylabel="Instantaneous Elo difference from winner's perspective",
    title="Markovich0104 vs Lagasuta — Performance vs Engine Evaluation",
  )
  engine_axis.set_ylabel("Engine evaluation from winner's perspective (pawns)")
  strength_limit = max(abs(value) for value in strength_axis.get_ylim())
  engine_limit = max(abs(value) for value in engine_axis.get_ylim())
  strength_axis.set_ylim(-strength_limit, strength_limit)
  engine_axis.set_ylim(-engine_limit, engine_limit)
  strength_axis.legend(
    handles=[strength_line, engine_line],
    loc="upper left",
    frameon=True,
  )
  figure.tight_layout()
  figure.savefig(output_path, dpi=180, bbox_inches="tight")
  print(
    f"Plot configuration: rows={len(data)}, max_ply={int(data['observed_plies'].max())}; "
    f"wrote {output_path}"
  )


if __name__ == "__main__":
  main()
