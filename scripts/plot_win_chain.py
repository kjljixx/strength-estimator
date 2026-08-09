#!/usr/bin/env python3
"""Plot win-chain QA figures from plot_data.json (written by sgf_filter_win_chain).

Safe to run mid-datagen: reads the latest atomically-flushed plot_data.json.
Uses seaborn + pandas; matplotlib only for save/annotations.
"""

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
  df = pd.DataFrame({"player": [x[0] for x in top], "count": [x[1] for x in top]})
  df = df.iloc[::-1].reset_index(drop=True)
  plt.figure(figsize=(8, max(4.0, 0.28 * len(df))))
  ax = sns.barplot(data=df, x="count", y="player", orient="h", color="#4c78a8")
  ax.set_xlabel("Appearances in accepted chains")
  ax.set_ylabel("")
  ax.set_title(f"Top {len(df)} players in chains")
  unique = pc.get("unique_players_in_chains", "?")
  total = pc.get("total_slot_appearances", "?")
  plt.figtext(0.99, 0.01, f"unique={unique} slot_apps={total}", ha="right", fontsize=8)
  sns.despine(left=True)
  savefig(out_path)
  return True


def plot_color_by_slot(data: dict, out_path: str) -> bool:
  rows = data.get("color_by_slot") or []
  if not rows or sum(sum(r) for r in rows) == 0:
    print("skip color_by_slot: no color data yet", flush=True)
    return False
  slots = []
  p_w = []
  se = []
  for i, (w, b) in enumerate(rows):
    n = w + b
    slots.append(i)
    if n <= 0:
      p_w.append(float("nan"))
      se.append(0.0)
    else:
      p = w / n
      p_w.append(p)
      se.append(math.sqrt(p * (1.0 - p) / n))
  df = pd.DataFrame({"slot": slots, "p_w": p_w, "se": se})
  plt.figure(figsize=(7, 4.5))
  ax = sns.lineplot(data=df, x="slot", y="p_w", marker="o", color="#4c78a8")
  ax.errorbar(
    df["slot"],
    df["p_w"],
    yerr=df["se"],
    fmt="none",
    ecolor="#4c78a8",
    capsize=3,
    label="P(W) +/- binomial SE",
  )
  ax.axhline(0.5, color="#bbbbbb", linestyle="--", linewidth=1)
  ax.set_ylim(0.0, 1.0)
  ax.set_xlabel("Chain slot (0=weakest)")
  ax.set_ylabel("Fraction White")
  ax.set_title("Training-game color balance by slot")
  ax.legend(loc="best")
  sns.despine()
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
  df = pd.DataFrame({
    "metric": ["chains", "attempts"],
    "value": [chains, attempts],
  })
  plt.figure(figsize=(6.5, 4.5))
  ax = sns.barplot(
    data=df,
    x="metric",
    y="value",
    hue="metric",
    palette={"chains": "#4c78a8", "attempts": "#f58518"},
    legend=False,
  )
  if max_chains > 0:
    ax.axhline(max_chains, color="#4c78a8", linestyle="--", linewidth=1, label=f"max_chains={max_chains}")
  if max_attempts > 0:
    ax.axhline(max_attempts, color="#f58518", linestyle=":", linewidth=1, label=f"max_attempts={max_attempts}")
  for patch, val in zip(ax.patches, df["value"]):
    ax.text(
      patch.get_x() + patch.get_width() / 2,
      patch.get_height(),
      f"{val}",
      ha="center",
      va="bottom",
      fontsize=9,
    )
  ax.set_xlabel("")
  ax.set_ylabel("Count")
  ax.set_title(f"Sampling yield (success rate={rate:.4f})")
  if max_chains > 0 or max_attempts > 0:
    ax.legend(loc="best")
  sns.despine()
  savefig(out_path)
  return True


def plot_elo_diff_vs_link(data: dict, out_path: str) -> bool:
  moments_list = data.get("elo_rel_by_link") or []
  if not moments_list or all(int(m.get("n", 0)) == 0 for m in moments_list):
    print("skip elo_diff_vs_link: no elo_rel_by_link yet", flush=True)
    return False
  rows = []
  for k, m in enumerate(moments_list):
    mean, sd = mean_sd(m)
    if mean is None:
      continue
    if k == 0:
      rows.append({"link": k, "mean_rel_elo": 0.0, "sd": 0.0})
    else:
      rows.append({"link": k, "mean_rel_elo": mean, "sd": sd if sd is not None else 0.0})
  if not rows:
    print("skip elo_diff_vs_link: empty series", flush=True)
    return False
  df = pd.DataFrame(rows)
  plt.figure(figsize=(7, 4.5))
  ax = sns.lineplot(data=df, x="link", y="mean_rel_elo", marker="o", color="#4c78a8")
  ax.errorbar(
    df["link"],
    df["mean_rel_elo"],
    yerr=df["sd"],
    fmt="none",
    ecolor="#4c78a8",
    capsize=3,
  )
  ax.axhline(0.0, color="#bbbbbb", linestyle="--", linewidth=1)
  ax.set_xlabel("Link # (0=weakest)")
  ax.set_ylabel("Mean Elo relative to link 0")
  n0 = int(moments_list[0].get("n", 0)) if moments_list else 0
  ax.set_title(f"Elo diff vs link (mean +/- SD, n={n0})")
  sns.despine()
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
  centers = [lo + (i + 0.5) * width for i in range(n_bins)]
  edges = [lo + i * width for i in range(n_bins + 1)]
  df = pd.DataFrame({"center": centers, "weight": counts})
  plt.figure(figsize=(7.5, 4.5))
  ax = sns.histplot(
    data=df,
    x="center",
    weights="weight",
    bins=edges,
    color=color,
    element="bars",
  )
  mean, sd = mean_sd(hist.get("moments") or {})
  if mean is not None:
    ax.axvline(mean, color="#e45756", linewidth=1.5, label=f"mean={mean:.1f}")
    if sd is not None and sd > 0:
      ax.axvspan(mean - sd, mean + sd, color="#e45756", alpha=0.2, label=f"+/-SD={sd:.2f}")
    ax.legend(loc="best")
  under = hist.get("underflow", 0)
  over = hist.get("overflow", 0)
  n = int((hist.get("moments") or {}).get("n", 0))
  ax.set_xlabel(xlabel)
  ax.set_ylabel("Count")
  ax.set_title(f"{title} (n={n}, under={under}, over={over})")
  sns.despine()
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
    "Chain-edge Elo diff (dest - src)",
    "dest_elo - src_elo",
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
    "|WR - BR|",
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
