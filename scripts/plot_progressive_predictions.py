#!/usr/bin/env python3
"""Plot winner-perspective progressive Elo differences from an exported CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


REQUIRED_COLUMNS = {
  "example_id",
  "observed_plies",
  "game_progress",
  "winner_elo_difference",
  "context_only_winner_elo_difference",
}


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description="Plot progressive predicted Elo difference from the winner's perspective",
  )
  parser.add_argument("csv_path", type=Path)
  parser.add_argument("--output-dir", type=Path, default=Path("progressive_plots"))
  parser.add_argument("--max-ply", type=int)
  parser.add_argument("--progress-bin-percent", type=int, default=5)
  parser.add_argument("--dpi", type=int, default=180)
  return parser


def save_figure(path: Path, dpi: int) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  plt.tight_layout()
  plt.savefig(path, dpi=dpi, bbox_inches="tight")
  plt.close()
  print(f"Wrote {path}")


def comparison_data(data: pd.DataFrame) -> pd.DataFrame:
  progressive = data[[
    "example_id",
    "observed_plies",
    "game_progress",
    "winner_elo_difference",
  ]].rename(columns={"winner_elo_difference": "predicted_elo_difference"})
  progressive["estimate"] = "Progressive"
  context = data[[
    "example_id",
    "observed_plies",
    "game_progress",
    "context_only_winner_elo_difference",
  ]].rename(
    columns={"context_only_winner_elo_difference": "predicted_elo_difference"},
  )
  context["estimate"] = "Context only"
  return pd.concat([progressive, context], ignore_index=True)


def plot_by_ply(data: pd.DataFrame, output_dir: Path, dpi: int) -> None:
  plt.figure(figsize=(10, 6))
  ax = sns.lineplot(
    data=comparison_data(data),
    x="observed_plies",
    y="predicted_elo_difference",
    hue="estimate",
    style="estimate",
    errorbar=("ci", 95),
  )
  ax.axhline(0, color="black", linestyle=":", linewidth=1)
  ax.set(
    xlabel="Observed plies",
    ylabel="Predicted Elo difference from winner's perspective",
    title="Predicted Elo Difference as Games Progress",
  )
  save_figure(output_dir / "winner_elo_difference_by_ply.png", dpi)


def plot_by_progress(
  data: pd.DataFrame,
  output_dir: Path,
  dpi: int,
  bin_percent: int,
) -> None:
  plotted = comparison_data(data)
  bin_width = bin_percent / 100
  plotted["game_progress_percent"] = (
    (plotted["game_progress"] / bin_width).round() * bin_width * 100
  ).clip(0, 100)
  plt.figure(figsize=(10, 6))
  ax = sns.lineplot(
    data=plotted,
    x="game_progress_percent",
    y="predicted_elo_difference",
    hue="estimate",
    style="estimate",
    errorbar=("ci", 95),
  )
  ax.axhline(0, color="black", linestyle=":", linewidth=1)
  ax.set(
    xlabel="Recorded game completed (%)",
    ylabel="Predicted Elo difference from winner's perspective",
    title="Predicted Elo Difference by Normalized Game Progress",
    xlim=(0, 100),
  )
  save_figure(output_dir / "winner_elo_difference_by_progress.png", dpi)


def plot_sample_counts(data: pd.DataFrame, output_dir: Path, dpi: int) -> None:
  counts = data.groupby("observed_plies", as_index=False)["example_id"].nunique()
  counts = counts.rename(columns={"example_id": "game_count"})
  plt.figure(figsize=(10, 4.5))
  ax = sns.lineplot(data=counts, x="observed_plies", y="game_count")
  ax.set(
    xlabel="Observed plies",
    ylabel="Decisive games",
    title="Sample Count by Observed Ply",
  )
  save_figure(output_dir / "sample_count_by_ply.png", dpi)


def main(argv: list[str] | None = None) -> int:
  args = build_parser().parse_args(argv)
  if args.max_ply is not None and args.max_ply < 0:
    raise ValueError("--max-ply must be non-negative")
  if args.progress_bin_percent <= 0 or args.progress_bin_percent > 100:
    raise ValueError("--progress-bin-percent must be in [1, 100]")

  data = pd.read_csv(args.csv_path)
  missing = REQUIRED_COLUMNS - set(data.columns)
  if missing:
    raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
  if data.empty:
    raise ValueError("CSV has no decisive-game prediction rows")
  if args.max_ply is not None:
    data = data[data["observed_plies"] <= args.max_ply]
  if data.empty:
    raise ValueError("no rows remain after applying --max-ply")

  sns.set_theme(style="whitegrid", context="notebook")
  print(
    f"Plotting {len(data)} rows from {data['example_id'].nunique()} decisive games; "
    f"maximum observed ply={int(data['observed_plies'].max())}"
  )
  plot_by_ply(data, args.output_dir, args.dpi)
  plot_by_progress(data, args.output_dir, args.dpi, args.progress_bin_percent)
  plot_sample_counts(data, args.output_dir, args.dpi)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
