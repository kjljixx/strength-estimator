#!/usr/bin/env python3
"""Evaluate a strength checkpoint on held-out Elo-bucket games: accuracy, signed Elo error, pairwise ordering."""

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

from overall import BUCKET_PATTERN, color_move_count, estimate_elo
from prediction_testing.sgf import move_count

RATING_PATTERNS = {"white": re.compile(r"WR\[(\d+)\]"), "black": re.compile(r"BR\[(\d+)\]")}
ROW_FIELDS = ("low", "high", "white_rating", "black_rating", "white_score", "white_moves", "black_score", "black_moves")


def bucket_paths(directory: Path) -> list[Path]:
  paths = [path for path in directory.glob("*.txt") if BUCKET_PATTERN.match(path.stem)]
  if not paths:
    raise ValueError(f"no LOW_HIGH.txt files in {directory}")
  return sorted(paths, key=lambda path: int(path.stem.split("_")[0]))


def score_directory(scorer, directory: Path, cache: Path) -> list[dict[str, float]]:
  if cache.exists():
    with cache.open(newline="", encoding="utf-8") as stream:
      rows = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)]
    print(f"Loaded {len(rows)} cached scored games from {cache}", flush=True)
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
          plies = move_count(sgf)
          if plies < 2:
            raise ValueError("game needs at least two plies")
          scores = scorer.score_sgf(sgf)
          rows.append({
            "low": low, "high": high,
            "white_rating": int(RATING_PATTERNS["white"].search(sgf).group(1)),
            "black_rating": int(RATING_PATTERNS["black"].search(sgf).group(1)),
            "white_score": float(scores["white"]), "white_moves": color_move_count(plies, "white"),
            "black_score": float(scores["black"]), "black_moves": color_move_count(plies, "black"),
          })
        except (ValueError, RuntimeError, AttributeError) as error:
          skipped += 1
          print(f"Skip {path.name}:{line_number}: {error}", file=sys.stderr)
    print(f"Scored {path.name}: {len(rows) - before} games", flush=True)
  print(f"Scored {len(rows)} games from {directory}, skipped {skipped}", flush=True)

  cache.parent.mkdir(parents=True, exist_ok=True)
  with cache.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=ROW_FIELDS)
    writer.writeheader()
    writer.writerows(rows)
  return rows


def build_calibration(rows: list[dict[str, float]]) -> list[dict[str, float | int]]:
  calibration = []
  for low in sorted({int(row["low"]) for row in rows}):
    group = [row for row in rows if row["low"] == low]
    high = int(group[0]["high"])
    moves = sum(row["white_moves"] + row["black_moves"] for row in group)
    total = sum(row["white_score"] * row["white_moves"] + row["black_score"] * row["black_moves"] for row in group)
    calibration.append({"low": low, "high": high, "center": (low + high) / 2,
                        "games": len(group), "moves": moves, "mean_score": total / moves})
  return calibration


def player_entries(rows: list[dict[str, float]], low: int) -> dict[str, np.ndarray]:
  group = [row for row in rows if row["low"] == low]
  return {
    "score": np.array([row[f"{color}_score"] for row in group for color in ("white", "black")]),
    "moves": np.array([row[f"{color}_moves"] for row in group for color in ("white", "black")], dtype=float),
    "rating": np.array([row[f"{color}_rating"] for row in group for color in ("white", "black")], dtype=float),
  }


def run_trials(rows, calibration, sizes, trials, rng) -> dict[tuple[int, int], dict[str, np.ndarray]]:
  bucket_means = np.array([float(row["mean_score"]) for row in calibration])
  lows = [int(row["low"]) for row in calibration]
  results = {}
  for low in lows:
    entries = player_entries(rows, low)
    for size in sizes:
      chosen = rng.integers(0, len(entries["score"]), size=(trials, size))
      weights = entries["moves"][chosen]
      scores = (entries["score"][chosen] * weights).sum(axis=1) / weights.sum(axis=1)
      predicted = np.abs(scores[:, None] - bucket_means[None, :]).argmin(axis=1)
      results[(size, low)] = {
        "score": scores,
        "true_elo": entries["rating"][chosen].mean(axis=1),
        "estimated_elo": np.array([estimate_elo(float(score), calibration)["estimated_elo"] for score in scores]),
        "predicted_index": predicted,
        "true_index": np.full(trials, lows.index(low)),
      }
  return results


def summarize(results, calibration, sizes):
  lows = [int(row["low"]) for row in calibration]
  overall, by_bucket = [], []
  for size in sizes:
    stacked = {key: np.concatenate([results[(size, low)][key] for low in lows]) for key in results[(size, lows[0])]}
    error = stacked["estimated_elo"] - stacked["true_elo"]
    index_gap = np.abs(stacked["predicted_index"] - stacked["true_index"])
    overall.append({
      "games_averaged": size,
      "accuracy_exact": float((index_gap == 0).mean()),
      "accuracy_within_one_bucket": float((index_gap <= 1).mean()),
      "mean_signed_error": float(error.mean()),
      "mean_absolute_error": float(np.abs(error).mean()),
      "rmse": float(np.sqrt((error ** 2).mean())),
      "slope_estimated_vs_true": float(np.polyfit(stacked["true_elo"], stacked["estimated_elo"], 1)[0]),
    })
    for low in lows:
      bucket = results[(size, low)]
      bucket_error = bucket["estimated_elo"] - bucket["true_elo"]
      by_bucket.append({
        "games_averaged": size, "bucket": f"{low}-{int(next(r['high'] for r in calibration if int(r['low']) == low))}",
        "mean_signed_error": float(bucket_error.mean()), "error_std": float(bucket_error.std()),
        "mean_absolute_error": float(np.abs(bucket_error).mean()),
        "accuracy_exact": float((bucket["predicted_index"] == bucket["true_index"]).mean()),
      })
  return overall, by_bucket


def pairwise_ordering(results, calibration, sizes):
  lows = [int(row["low"]) for row in calibration]
  width = int(calibration[0]["high"]) - int(calibration[0]["low"])
  rows = []
  for size in sizes:
    for gap in range(1, len(lows)):
      correct = [float((results[(size, lows[i + gap])]["score"] > results[(size, lows[i])]["score"]).mean())
                 for i in range(len(lows) - gap)]
      rows.append({"games_averaged": size, "bucket_gap": gap, "elo_gap": gap * width,
                   "ordering_accuracy": float(np.mean(correct))})
  return rows


def calibration_monotonic(calibration) -> list[str]:
  return [f"{a['low']}-{a['high']} ({a['mean_score']:.4f}) >= {b['low']}-{b['high']} ({b['mean_score']:.4f})"
          for a, b in zip(calibration, calibration[1:]) if float(a["mean_score"]) >= float(b["mean_score"])]


def write_csv(path: Path, rows: list[dict]) -> None:
  with path.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)


def markdown_table(rows: list[dict]) -> str:
  headers = list(rows[0].keys())
  lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
  for row in rows:
    lines.append("| " + " | ".join(f"{value:.3f}" if isinstance(value, float) else str(value) for value in row.values()) + " |")
  return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--config", type=Path, required=True)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--candidate-dir", type=Path, required=True)
  parser.add_argument("--query-dir", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--sizes", type=int, nargs="+", default=[1, 10, 25, 50, 75, 100])
  parser.add_argument("--trials", type=int, default=500, help="random game groups per bucket and size")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--gpu-id", type=int, default=0)
  return parser


def main() -> int:
  args = build_parser().parse_args()
  print(f"Config: {vars(args)}", flush=True)
  from build.chess import strength_py

  scorer = strength_py.StrengthScorer(str(args.config), str(args.checkpoint), args.gpu_id)
  if strength_py.get_bt_use_weight():
    raise ValueError("standard-compatible aggregation requires bt_use_weight=false")
  args.output_dir.mkdir(parents=True, exist_ok=True)

  candidate_rows = score_directory(scorer, args.candidate_dir, args.output_dir / "candidate_scores.csv")
  query_rows = score_directory(scorer, args.query_dir, args.output_dir / "query_scores.csv")
  calibration = build_calibration(candidate_rows)
  (args.output_dir / "calibration.json").write_text(json.dumps(calibration, indent=2), encoding="utf-8")
  violations = calibration_monotonic(calibration)
  print(f"Calibration: {len(calibration)} buckets, order violations: {violations or 'none'}", flush=True)
  if {int(row["low"]) for row in query_rows} != {int(row["low"]) for row in calibration}:
    raise ValueError("candidate and query directories have different buckets")

  rng = np.random.default_rng(args.seed)
  results = run_trials(query_rows, calibration, args.sizes, args.trials, rng)
  overall, by_bucket = summarize(results, calibration, args.sizes)
  pairwise = pairwise_ordering(results, calibration, args.sizes)
  write_csv(args.output_dir / "overall_by_games.csv", overall)
  write_csv(args.output_dir / "bucket_by_games.csv", by_bucket)
  write_csv(args.output_dir / "pairwise_ordering.csv", pairwise)

  report = [
    f"# Evaluation: {args.checkpoint}", "",
    f"Config `{args.config}`, candidates `{args.candidate_dir}`, queries `{args.query_dir}`, "
    f"{args.trials} random groups per bucket and size, seed {args.seed}.",
    "Signed error = estimated Elo - mean recorded rating of the sampled players.", "",
    f"Calibration order violations: {violations or 'none'}", "",
    "## Overall", markdown_table(overall), "",
  ]
  for size in args.sizes:
    report += [f"## Per bucket, {size} games averaged", markdown_table([row for row in by_bucket if row["games_averaged"] == size]), ""]
  for size in args.sizes:
    report += [f"## Pairwise ordering, {size} games averaged", markdown_table([row for row in pairwise if row["games_averaged"] == size]), ""]
  (args.output_dir / "results.md").write_text("\n".join(report), encoding="utf-8")
  for row in overall:
    print(json.dumps(row), flush=True)
  print(f"Wrote results to {args.output_dir}", flush=True)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
