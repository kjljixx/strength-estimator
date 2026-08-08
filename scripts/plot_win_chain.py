#!/usr/bin/env python3
"""Plot win-chain QA figures from plot_data.json (written by sgf_filter_win_chain).

Safe to run mid-datagen: reads the latest atomically-flushed plot_data.json.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt


def mean_sd(moments: dict) -> Tuple[Optional[float], Optional[float]]:
  """Return (mean, sample SD). Sample SD uses n-1 in the denominator."""
  n = int(moments.get("n", 0))
  if n <= 0:
    return None, None
  mean = float(moments["sum"]) / n
  if n < 2:
    return mean, 0.0
  # sample variance from online moments: (sum_sq - n*mean^2) / (n-1)
  ss = float(moments["sum_sq"]) - n * mean * mean
  if ss < 0:
    ss = 0.0
  sd = math.sqrt(ss / (n - 1))
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
  pc = data.get("player_counts") or {}
  top = pc.get("top") or []
  if not top:
    print("skip player_freq: no player_counts yet", flush=True)
    return False
  top = top[:top_n]
  names = [x[0] for x in top][::-1]
  counts = [x[1] for x in top][::-1]
  plt.figure(figsize=(8, max(4.0, 0.28 * len(names))))
  plt.barh(names, counts, color="#4c78a8")
  plt.xlabel("Appearances in accepted chains")
  plt.title(f"Top {len(names)} players in chains")
  unique = pc.get("unique_players_in_chains", "?")
  total = pc.get("total_slot_appearances", "?")
  plt.figtext(0.99, 0.01, f"unique={unique} slot_apps={total}", ha="right", fontsize=8)
  savefig(out_path)
  return True


def plot_color_by_slot(data: dict, out_path: str) -> bool:
  rows = data.get("color_by_slot") or []
  if not rows or sum(sum(r) for r in rows) == 0:
    print("skip color_by_slot: no color data yet", flush=True)
    return False
  slots = list(range(len(rows)))
  p_w = []
  se = []
  ns = []
  for w, b in rows:
    n = w + b
    ns.append(n)
    if n <= 0:
      p_w.append(float("nan"))
      se.append(0.0)
    else:
      p = w / n
      p_w.append(p)
      se.append(math.sqrt(p * (1.0 - p) / n))
  plt.figure(figsize=(7, 4.5))
  plt.errorbar(slots, p_w, yerr=se, fmt="-o", capsize=3, color="#4c78a8", label="P(W) +/- binomial SE")
  plt.axhline(0.5, color="#bbbbbb", linestyle="--", linewidth=1)
  plt.ylim(0.0, 1.0)
  plt.xlabel("Chain slot (0=weakest)")
  plt.ylabel("Fraction White")
  plt.title("Training-game color balance by slot")
  plt.legend(loc="best")
  savefig(out_path)
  return True


def plot_sampling_yield(data: dict, out_path: str) -> bool:
  s = data.get("sampling") or {}
  chains = int(s.get("chains", 0))
  attempts = int(s.get("attempts", 0))
  max_chains = int(s.get("max_chains", 0))
  max_attempts = int(s.get("max_attempts", 0))
  if attempts <= 0 and chains <= 0:
    print("skip sampling_yield: no sampling yet", flush=True)
    return False
  rate = (chains / attempts) if attempts > 0 else 0.0
  plt.figure(figsize=(6.5, 4.5))
  labels = ["chains", "attempts"]
  values = [chains, attempts]
  colors = ["#4c78a8", "#f58518"]
  bars = plt.bar(labels, values, color=colors)
  if max_chains > 0:
    plt.axhline(max_chains, color="#4c78a8", linestyle="--", linewidth=1, label=f"max_chains={max_chains}")
  if max_attempts > 0:
    plt.axhline(max_attempts, color="#f58518", linestyle=":", linewidth=1, label=f"max_attempts={max_attempts}")
  for bar, val in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val}", ha="center", va="bottom", fontsize=9)
  plt.ylabel("Count")
  plt.title(f"Sampling yield (success rate={rate:.4f})")
  if max_chains > 0 or max_attempts > 0:
    plt.legend(loc="best")
  savefig(out_path)
  return True


def plot_elo_diff_vs_link(data: dict, out_path: str) -> bool:
  moments_list = data.get("elo_rel_by_link") or []
  if not moments_list or all(int(m.get("n", 0)) == 0 for m in moments_list):
    print("skip elo_diff_vs_link: no elo_rel_by_link yet", flush=True)
    return False
  xs = []
  ys = []
  yerr = []
  for k, m in enumerate(moments_list):
    mean, sd = mean_sd(m)
    if mean is None:
      continue
    xs.append(k)
    if k == 0:
      ys.append(0.0)
      yerr.append(0.0)
    else:
      ys.append(mean)
      yerr.append(sd if sd is not None else 0.0)
  if not xs:
    print("skip elo_diff_vs_link: empty series", flush=True)
    return False
  plt.figure(figsize=(7, 4.5))
  plt.errorbar(xs, ys, yerr=yerr, fmt="-o", capsize=3, color="#4c78a8")
  plt.axhline(0.0, color="#bbbbbb", linestyle="--", linewidth=1)
  plt.xlabel("Link # (0=weakest)")
  plt.ylabel("Mean Elo relative to link 0")
  n0 = int(moments_list[0].get("n", 0)) if moments_list else 0
  plt.title(f"Elo diff vs link (mean +/- SD, n={n0})")
  savefig(out_path)
  return True


def _plot_hist_with_mean(
  hist: dict,
  out_path: str,
  title: str,
  xlabel: str,
  color: str,
) -> bool:
  counts = hist.get("counts") or []
  if not counts or sum(counts) == 0:
    return False
  lo = float(hist["lo"])
  hi = float(hist["hi"])
  n_bins = int(hist["n_bins"])
  width = (hi - lo) / n_bins
  edges = [lo + i * width for i in range(n_bins + 1)]
  centers = [0.5 * (edges[i] + edges[i + 1]) for i in range(n_bins)]
  plt.figure(figsize=(7.5, 4.5))
  plt.bar(centers, counts, width=width * 0.92, color=color, align="center")
  mean, sd = mean_sd(hist.get("moments") or {})
  if mean is not None:
    plt.axvline(mean, color="#e45756", linewidth=1.5, label=f"mean={mean:.1f}")
    if sd is not None and sd > 0:
      plt.axvspan(mean - sd, mean + sd, color="#e45756", alpha=0.2, label=f"+/-SD={sd:.2f}")
    plt.legend(loc="best")
  under = hist.get("underflow", 0)
  over = hist.get("overflow", 0)
  n = int((hist.get("moments") or {}).get("n", 0))
  plt.xlabel(xlabel)
  plt.ylabel("Count")
  plt.title(f"{title} (n={n}, under={under}, over={over})")
  savefig(out_path)
  return True


def plot_chain_edge_elo_diff(data: dict, out_path: str) -> bool:
  hist = data.get("chain_edge_elo_diffs") or {}
  if not hist or int((hist.get("moments") or {}).get("n", 0)) == 0:
    print("skip chain_edge_elo_diff: no edge diffs yet", flush=True)
    return False
  ok = _plot_hist_with_mean(
    hist,
    out_path,
    "Chain-edge Elo diff (dest − src)",
    "dest_elo − src_elo",
    "#54a24b",
  )
  if not ok:
    print("skip chain_edge_elo_diff: empty hist", flush=True)
  return ok


def plot_abs_elo_diff_all(data: dict, out_path: str) -> bool:
  hist = data.get("abs_elo_diff_all") or {}
  if not hist or int((hist.get("moments") or {}).get("n", 0)) == 0:
    print("skip abs_elo_diff_all: no pass1 abs diffs yet", flush=True)
    return False
  ok = _plot_hist_with_mean(
    hist,
    out_path,
    "Abs Elo diff of all pass1 games",
    "|WR − BR|",
    "#b279a2",
  )
  if not ok:
    print("skip abs_elo_diff_all: empty hist", flush=True)
  return ok


def print_summary(data: dict) -> None:
  meta = data.get("meta") or {}
  s = data.get("sampling") or {}
  edge_m = (data.get("chain_edge_elo_diffs") or {}).get("moments") or {}
  abs_m = (data.get("abs_elo_diff_all") or {}).get("moments") or {}
  edge_mean, edge_sd = mean_sd(edge_m)
  abs_mean, abs_sd = mean_sd(abs_m)
  chains = s.get("chains", 0)
  attempts = s.get("attempts", 0)
  rate = (chains / attempts) if attempts else 0.0
  edge_s = "n/a" if edge_mean is None else f"{edge_mean:.2f}+/-{edge_sd:.2f} SD"
  abs_s = "n/a" if abs_mean is None else f"{abs_mean:.2f}+/-{abs_sd:.2f} SD"
  print(
    f"summary phase={meta.get('phase')} chains={chains}/{s.get('max_chains')} "
    f"attempts={attempts} success={rate:.4f} "
    f"edge_dElo={edge_s} abs_dElo={abs_s} "
    f"edge_elo_skip={data.get('edge_elo_skip', 0)}",
    flush=True,
  )


def main() -> None:
  parser = argparse.ArgumentParser(description="Plot win-chain QA figures from plot_data.json")
  parser.add_argument("--data-dir", default="training_sgf_chess_chain")
  parser.add_argument("--plot-data", default="", help="Default: {data-dir}/plot_data.json")
  parser.add_argument("--output-dir", default="", help="Default: {data-dir}/figures")
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
