#!/usr/bin/env python3
"""Plot win-chain QA figures from plot_data.json (sgf_filter_win_chain sidecar)."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme()


def mean_sd(moments: dict) -> Tuple[Optional[float], Optional[float]]:
  n = int(moments.get("n", 0))
  if n <= 0:
    return None, None
  mean = float(moments["sum"]) / n
  if n < 2:
    return mean, 0.0
  ss = float(moments["sum_sq"]) - n * mean * mean
  sd = math.sqrt(max(ss, 0.0) / (n - 1))
  return mean, sd


def load_plot_data(path: str) -> dict:
  with open(path, "r", encoding="utf-8") as f:
    return json.load(f)


def savefig(path: str) -> None:
  os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
  plt.tight_layout()
  plt.savefig(path, dpi=160)
  plt.close()
  print(f"wrote {path}", flush=True)


def plot_player_freq(data: dict, out_path: str, top_n: int = 30) -> bool:
  top = (data.get("player_counts") or {}).get("top") or []
  if not top:
    print("skip player_freq: no data", flush=True)
    return False
  df = pd.DataFrame(top[:top_n], columns=["player", "count"]).iloc[::-1]
  plt.figure(figsize=(8, max(4.0, 0.28 * len(df))))
  ax = sns.barplot(data=df, x="count", y="player", orient="h")
  ax.set(xlabel="Appearances in accepted chains", ylabel="", title=f"Top {len(df)} players")
  savefig(out_path)
  return True


def plot_color_by_slot(data: dict, out_path: str) -> bool:
  rows = data.get("color_by_slot") or []
  if not rows or sum(sum(r) for r in rows) == 0:
    print("skip color_by_slot: no data", flush=True)
    return False
  records = []
  for slot, (w, b) in enumerate(rows):
    n = w + b
    p = w / n if n else float("nan")
    se = math.sqrt(p * (1.0 - p) / n) if n else 0.0
    records.append({"slot": slot, "p_w": p, "se": se})
  df = pd.DataFrame(records)
  plt.figure(figsize=(7, 4.5))
  ax = sns.lineplot(data=df, x="slot", y="p_w", marker="o")
  ax.errorbar(df["slot"], df["p_w"], yerr=df["se"], fmt="none", capsize=3)
  ax.axhline(0.5, linestyle="--", linewidth=1)
  ax.set(ylim=(0, 1), xlabel="Chain slot (0=weakest)", ylabel="Fraction White",
         title="Color balance by slot (P(W) +/- binomial SE)")
  savefig(out_path)
  return True


def plot_sampling_yield(data: dict, out_path: str) -> bool:
  s = data.get("sampling") or {}
  chains, attempts = int(s.get("chains", 0)), int(s.get("attempts", 0))
  if chains <= 0 and attempts <= 0:
    print("skip sampling_yield: no data", flush=True)
    return False
  rate = chains / attempts if attempts else 0.0
  df = pd.DataFrame({"metric": ["chains", "attempts"], "value": [chains, attempts]})
  plt.figure(figsize=(6.5, 4.5))
  ax = sns.barplot(data=df, x="metric", y="value", hue="metric", legend=False)
  for max_key, style in (("max_chains", "--"), ("max_attempts", ":")):
    m = int(s.get(max_key, 0))
    if m > 0:
      ax.axhline(m, linestyle=style, linewidth=1, label=f"{max_key}={m}")
  ax.set(xlabel="", ylabel="Count", title=f"Sampling yield (success={rate:.4f})")
  if any(int(s.get(k, 0)) > 0 for k in ("max_chains", "max_attempts")):
    ax.legend(loc="best")
  savefig(out_path)
  return True


def plot_elo_diff_vs_link(data: dict, out_path: str) -> bool:
  moments_list = data.get("elo_rel_by_link") or []
  if not moments_list or all(int(m.get("n", 0)) == 0 for m in moments_list):
    print("skip elo_diff_vs_link: no data", flush=True)
    return False
  rows = []
  for k, m in enumerate(moments_list):
    mean, sd = mean_sd(m)
    if mean is None:
      continue
    rows.append({"link": k, "mean_rel_elo": 0.0 if k == 0 else mean, "sd": 0.0 if k == 0 else sd})
  if not rows:
    print("skip elo_diff_vs_link: empty", flush=True)
    return False
  df = pd.DataFrame(rows)
  plt.figure(figsize=(7, 4.5))
  ax = sns.lineplot(data=df, x="link", y="mean_rel_elo", marker="o")
  ax.errorbar(df["link"], df["mean_rel_elo"], yerr=df["sd"], fmt="none", capsize=3)
  ax.axhline(0.0, linestyle="--", linewidth=1)
  n0 = int(moments_list[0].get("n", 0))
  ax.set(xlabel="Link # (0=weakest)", ylabel="Mean Elo relative to link 0",
         title=f"Elo diff vs link (mean +/- SD, n={n0})")
  savefig(out_path)
  return True


def _plot_hist_with_mean(hist: dict, out_path: str, title: str, xlabel: str) -> bool:
  counts = hist.get("counts") or []
  if not counts or sum(counts) == 0:
    return False
  lo, hi, n_bins = float(hist["lo"]), float(hist["hi"]), int(hist["n_bins"])
  width = (hi - lo) / n_bins
  centers = [lo + (i + 0.5) * width for i in range(n_bins)]
  edges = [lo + i * width for i in range(n_bins + 1)]
  df = pd.DataFrame({"center": centers, "weight": counts})
  plt.figure(figsize=(7.5, 4.5))
  ax = sns.histplot(data=df, x="center", weights="weight", bins=edges)
  mean, sd = mean_sd(hist.get("moments") or {})
  if mean is not None:
    ax.axvline(mean, linewidth=1.5, label=f"mean={mean:.1f}")
    if sd and sd > 0:
      ax.axvspan(mean - sd, mean + sd, alpha=0.2, label=f"+/-SD={sd:.2f}")
    ax.legend(loc="best")
  n = int((hist.get("moments") or {}).get("n", 0))
  ax.set(xlabel=xlabel, ylabel="Count",
         title=f"{title} (n={n}, under={hist.get('underflow', 0)}, over={hist.get('overflow', 0)})")
  savefig(out_path)
  return True


def plot_chain_edge_elo_diff(data: dict, out_path: str) -> bool:
  hist = data.get("chain_edge_elo_diffs") or {}
  if not hist or int((hist.get("moments") or {}).get("n", 0)) == 0:
    print("skip chain_edge_elo_diff: no data", flush=True)
    return False
  return _plot_hist_with_mean(hist, out_path, "Chain-edge Elo diff (dest - src)", "dest_elo - src_elo")


def plot_abs_elo_diff_all(data: dict, out_path: str) -> bool:
  hist = data.get("abs_elo_diff_all") or {}
  if not hist or int((hist.get("moments") or {}).get("n", 0)) == 0:
    print("skip abs_elo_diff_all: no data", flush=True)
    return False
  return _plot_hist_with_mean(hist, out_path, "Abs Elo diff of all pass1 games", "|WR - BR|")


def print_summary(data: dict) -> None:
  s = data.get("sampling") or {}
  edge_mean, edge_sd = mean_sd((data.get("chain_edge_elo_diffs") or {}).get("moments") or {})
  abs_mean, abs_sd = mean_sd((data.get("abs_elo_diff_all") or {}).get("moments") or {})
  chains, attempts = s.get("chains", 0), s.get("attempts", 0)
  rate = chains / attempts if attempts else 0.0
  edge_s = "n/a" if edge_mean is None else f"{edge_mean:.2f}+/-{edge_sd:.2f} SD"
  abs_s = "n/a" if abs_mean is None else f"{abs_mean:.2f}+/-{abs_sd:.2f} SD"
  print(
    f"summary phase={(data.get('meta') or {}).get('phase')} "
    f"chains={chains}/{s.get('max_chains')} attempts={attempts} success={rate:.4f} "
    f"edge_dElo={edge_s} abs_dElo={abs_s} edge_elo_skip={data.get('edge_elo_skip', 0)}",
    flush=True,
  )


def main() -> None:
  parser = argparse.ArgumentParser(description="Plot win-chain QA figures from plot_data.json")
  parser.add_argument("--data-dir", default="training_sgf_chess_chain")
  parser.add_argument("--plot-data", default="")
  parser.add_argument("--output-dir", default="")
  parser.add_argument("--top-players", type=int, default=30)
  args = parser.parse_args()

  plot_data_path = args.plot_data or os.path.join(args.data_dir, "plot_data.json")
  out_dir = args.output_dir or os.path.join(args.data_dir, "figures")
  if not os.path.isfile(plot_data_path):
    print(f"ERROR: plot data not found: {plot_data_path}", flush=True)
    sys.exit(1)

  print(f"loading {plot_data_path}", flush=True)
  data = load_plot_data(plot_data_path)
  print_summary(data)
  os.makedirs(out_dir, exist_ok=True)

  plot_player_freq(data, os.path.join(out_dir, "player_freq.png"), top_n=args.top_players)
  plot_color_by_slot(data, os.path.join(out_dir, "color_by_slot.png"))
  plot_sampling_yield(data, os.path.join(out_dir, "sampling_yield.png"))
  plot_elo_diff_vs_link(data, os.path.join(out_dir, "elo_diff_vs_link.png"))
  plot_chain_edge_elo_diff(data, os.path.join(out_dir, "chain_edge_elo_diff.png"))
  plot_abs_elo_diff_all(data, os.path.join(out_dir, "abs_elo_diff_all.png"))


if __name__ == "__main__":
  main()
