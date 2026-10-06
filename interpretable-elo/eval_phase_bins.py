#!/usr/bin/env python3
"""Per-bin phase Elo gaps: each phase's Elo estimate minus the average of the three phase estimates."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval_bins import RATING_PATTERNS, bucket_paths, markdown_table, monotone_elo_curve, write_csv
from phase_elo import phase_totals
from phase_rules import PHASES, boards_after_moves

MOVE_PATTERN = re.compile(r";[BW]\[([a-h][1-8][a-h][1-8][qrbn]?)\]")
COLORS = ("white", "black")
PHASE_FIELDS = tuple(f"{color}_{phase}_{kind}" for color in COLORS for phase in PHASES for kind in ("sum", "count"))
ROW_FIELDS = ("low", "high", "white_rating", "black_rating") + PHASE_FIELDS


def score_phase_games(scorer, directory: Path, cache: Path) -> list[dict[str, float]]:
  if cache.exists():
    with cache.open(newline="", encoding="utf-8") as stream:
      rows = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)]
    print(f"Loaded {len(rows)} cached phase-scored games from {cache}", flush=True)
    return rows

  rows: list[dict[str, float]] = []
  skipped = 0
  for path in bucket_paths(directory):
    low, high = map(int, path.stem.split("_"))
    before = len(rows)
    with path.open(encoding="utf-8", errors="replace") as stream:
      for line_number, sgf in enumerate(stream, start=1):
        if not sgf.strip():
          continue
        try:
          boards = boards_after_moves(MOVE_PATTERN.findall(sgf))
          if len(boards) < 2:
            raise ValueError("game needs at least two plies")
          totals = phase_totals(scorer, sgf, boards)
          row = {"low": low, "high": high,
                 "white_rating": int(RATING_PATTERNS["white"].search(sgf).group(1)),
                 "black_rating": int(RATING_PATTERNS["black"].search(sgf).group(1))}
          for color in COLORS:
            for phase in PHASES:
              row[f"{color}_{phase}_sum"], row[f"{color}_{phase}_count"] = totals[color][phase]
          rows.append(row)
        except (ValueError, RuntimeError, AttributeError) as error:
          skipped += 1
          print(f"Skip {path.name}:{line_number}: {error}", file=sys.stderr)
    print(f"Phase-scored {path.name}: {len(rows) - before} games", flush=True)
  print(f"Phase-scored {len(rows)} games from {directory}, skipped {skipped}", flush=True)

  cache.parent.mkdir(parents=True, exist_ok=True)
  with cache.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=ROW_FIELDS)
    writer.writeheader()
    writer.writerows(rows)
  return rows


def phase_estimates(sums: np.ndarray, counts: np.ndarray, curve: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
  """Elo estimate per phase from summed scores and move counts of shape (games, 3)."""
  return np.interp(sums.sum(axis=0) / counts.sum(axis=0), *curve)


def phase_gaps(rows, curve, bootstrap: int, rng) -> list[dict]:
  results = []
  for low in sorted({int(row["low"]) for row in rows}):
    group = [row for row in rows if row["low"] == low]
    sums = np.array([[row["white_%s_sum" % phase] + row["black_%s_sum" % phase] for phase in PHASES] for row in group])
    counts = np.array([[row["white_%s_count" % phase] + row["black_%s_count" % phase] for phase in PHASES] for row in group])
    true_rating = float(np.mean([[row["white_rating"], row["black_rating"]] for row in group]))
    estimates = phase_estimates(sums, counts, curve)
    gaps = estimates - estimates.mean()
    resampled = np.array([
      (lambda e: e - e.mean())(phase_estimates(sums[picks], counts[picks], curve))
      for picks in rng.integers(0, len(group), size=(bootstrap, len(group)))])
    for index, phase in enumerate(PHASES):
      results.append({
        "bucket": f"{low}-{int(group[0]['high'])}", "phase": phase, "games": len(group),
        "moves": int(counts[:, index].sum()),
        "mean_phase_score": float(sums[:, index].sum() / counts[:, index].sum()),
        "estimated_elo": float(estimates[index]), "true_rating": true_rating,
        "error_vs_true_rating": float(estimates[index] - true_rating),
        "gap_vs_phase_average": float(gaps[index]), "gap_se": float(resampled[:, index].std()),
      })
  for phase in PHASES:
    phase_rows = [row for row in results if row["phase"] == phase]
    offset = float(np.mean([row["gap_vs_phase_average"] for row in phase_rows]))
    for row in phase_rows:
      row["gap_centered"] = row["gap_vs_phase_average"] - offset
  return results


def pivot(results: list[dict], key: str, with_se: bool = False) -> list[dict]:
  rows = []
  for bucket in dict.fromkeys(row["bucket"] for row in results):
    row = {"bucket": bucket}
    for phase in PHASES:
      cell = next(r for r in results if r["bucket"] == bucket and r["phase"] == phase)
      row[phase] = f"{cell[key]:+.1f} ± {cell['gap_se']:.1f}" if with_se else f"{cell[key]:+.1f}"
    rows.append(row)
  return rows


def summary(results: list[dict]) -> list[dict]:
  return [{"phase": phase,
           "mean_gap": float(np.mean([r["gap_vs_phase_average"] for r in results if r["phase"] == phase])),
           "mean_abs_gap": float(np.mean([abs(r["gap_vs_phase_average"]) for r in results if r["phase"] == phase])),
           "mean_error_vs_true_rating": float(np.mean([r["error_vs_true_rating"] for r in results if r["phase"] == phase]))}
          for phase in PHASES]


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--config", type=Path, required=True)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--calibration", type=Path, required=True, help="whole-game calibration.json from eval_bins.py")
  parser.add_argument("--query-dir", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--bootstrap", type=int, default=200)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--gpu-id", type=int, default=0)
  return parser


def main() -> int:
  args = build_parser().parse_args()
  print(f"Config: {vars(args)}", flush=True)
  from build.chess import strength_py

  scorer = strength_py.StrengthScorer(str(args.config), str(args.checkpoint), args.gpu_id)
  args.output_dir.mkdir(parents=True, exist_ok=True)
  calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
  curve = monotone_elo_curve(calibration)

  rows = score_phase_games(scorer, args.query_dir, args.output_dir / "phase_query_scores.csv")
  results = phase_gaps(rows, curve, args.bootstrap, np.random.default_rng(args.seed))
  write_csv(args.output_dir / "phase_gaps.csv", results)
  report = [
    f"# Phase Elo gaps: {args.checkpoint}", "",
    f"Queries `{args.query_dir}`, calibration `{args.calibration}` (whole-game, monotone, clamped), {args.bootstrap} bootstrap resamples over games.",
    "Phase Elo = Elo of the mean score over all moves in that phase, pooled over the bin's games.",
    "Gap = phase Elo minus the average of the bin's three phase Elos (sums to zero within a bin); ± is the bootstrap standard error.",
    "Centered gap subtracts each phase's mean gap over bins, leaving only how the gap changes with rating.",
    "Error vs true rating = phase Elo minus the mean recorded rating of the bin's players.", "",
    "## Summary over bins", markdown_table(summary(results)), "",
    "## Gap vs phase average", markdown_table(pivot(results, "gap_vs_phase_average", with_se=True)), "",
    "## Centered gap", markdown_table(pivot(results, "gap_centered")), "",
    "## Error vs true rating", markdown_table(pivot(results, "error_vs_true_rating")), "",
  ]
  (args.output_dir / "phase_results.md").write_text("\n".join(report), encoding="utf-8")
  for row in summary(results):
    print(json.dumps(row), flush=True)
  print(f"Wrote phase results to {args.output_dir}", flush=True)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
