#!/usr/bin/env python3
"""Subsample win-chain data by keeping every Nth slot of each chain.

Reads an sgf_filter_win_chain output folder (games.txt + chains.txt) and writes
another folder in the same format. For each chain, keeps slots at indices
0, stride, 2*stride, ... then remaps games.txt to only referenced games.

Training note: set bt_num_rank_per_batch (and learner_batch_size /
nn_rank_size) to the new chain length after striding.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from typing import Dict, List, Optional, Tuple

Slot = Tuple[int, str]


def log(msg: str) -> None:
  print(msg, flush=True)


def parse_chain_line(line: str, line_no: int) -> Optional[List[Slot]]:
  line = line.strip()
  if not line:
    return None
  slots: List[Slot] = []
  for token in line.split():
    colon = token.find(":")
    if colon < 1 or colon + 1 >= len(token):
      log(f"WARN: bad token {token!r} at chains.txt line {line_no}")
      return None
    try:
      game_id = int(token[:colon])
    except ValueError:
      log(f"WARN: bad game id in {token!r} at chains.txt line {line_no}")
      return None
    color = token[colon + 1]
    if color not in ("W", "B", "w", "b"):
      log(f"WARN: bad color in {token!r} at chains.txt line {line_no}")
      return None
    slots.append((game_id, color.upper()))
  return slots if slots else None


def stride_chain(slots: List[Slot], stride: int) -> List[Slot]:
  return slots[::stride]


def main() -> None:
  parser = argparse.ArgumentParser(
    description="Keep every Nth slot of each win-chain; rewrite games/chains pool"
  )
  parser.add_argument("--input-dir", default="training_sgf_chess_chain")
  parser.add_argument(
    "--output-dir",
    default=None,
    help="Output folder (default: {input-dir}_s{stride})",
  )
  parser.add_argument("--stride", type=int, required=True, help="Keep indices 0,stride,2*stride,... (>=2)")
  args = parser.parse_args()

  if args.stride < 2:
    log("ERROR: --stride must be >= 2")
    sys.exit(1)

  input_dir = args.input_dir
  output_dir = args.output_dir
  if output_dir is None:
    output_dir = f"{input_dir.rstrip('/').rstrip(os.sep)}_s{args.stride}"

  games_in = os.path.join(input_dir, "games.txt")
  chains_in = os.path.join(input_dir, "chains.txt")
  if not os.path.isfile(games_in):
    log(f"ERROR: missing {games_in}")
    sys.exit(1)
  if not os.path.isfile(chains_in):
    log(f"ERROR: missing {chains_in}")
    sys.exit(1)

  log("=== sgf_win_chain_stride config ===")
  log(f"input_dir={input_dir} output_dir={output_dir} stride={args.stride}")

  kept_chains: List[List[Slot]] = []
  input_chains = 0
  dropped_short = 0
  parse_skip = 0
  in_len_hist: Counter = Counter()
  out_len_hist: Counter = Counter()

  log(f"reading chains {chains_in}")
  with open(chains_in, "r", encoding="utf-8") as f:
    for line_no, line in enumerate(f, start=1):
      slots = parse_chain_line(line, line_no)
      if slots is None:
        if line.strip():
          parse_skip += 1
        continue
      input_chains += 1
      in_len_hist[len(slots)] += 1
      strided = stride_chain(slots, args.stride)
      if len(strided) < 2:
        dropped_short += 1
        continue
      out_len_hist[len(strided)] += 1
      kept_chains.append(strided)

  if not kept_chains:
    log("ERROR: no chains kept after stride")
    sys.exit(1)

  used_old: set = set()
  for chain in kept_chains:
    for game_id, _ in chain:
      used_old.add(game_id)

  # Dense remap in ascending old-id order so games.txt line order is stable.
  old_sorted = sorted(used_old)
  old_to_new: Dict[int, int] = {old: new for new, old in enumerate(old_sorted)}
  used_set = set(old_sorted)

  os.makedirs(output_dir, exist_ok=True)
  games_out = os.path.join(output_dir, "games.txt")
  chains_out = os.path.join(output_dir, "chains.txt")

  log(f"collecting {len(old_sorted)} referenced games from {games_in}")
  kept_lines: Dict[int, str] = {}
  with open(games_in, "r", encoding="utf-8") as fin:
    for line_idx, line in enumerate(fin):
      if line_idx not in used_set:
        continue
      if not line.endswith("\n"):
        line += "\n"
      kept_lines[line_idx] = line

  missing = [old for old in old_sorted if old not in kept_lines]
  if missing:
    log(f"ERROR: games.txt missing {len(missing)} referenced ids (e.g. {missing[:5]})")
    sys.exit(1)

  log(f"writing {len(old_sorted)} games -> {games_out}")
  with open(games_out, "w", encoding="utf-8") as fout:
    for i, old in enumerate(old_sorted):
      fout.write(kept_lines[old])
      if (i + 1) % 50_000 == 0:
        log(f"wrote games {i + 1}/{len(old_sorted)}")
  written = len(old_sorted)

  log(f"writing {len(kept_chains)} chains -> {chains_out}")
  with open(chains_out, "w", encoding="utf-8") as out:
    for chain in kept_chains:
      out.write(
        " ".join(f"{old_to_new[gid]}:{color}" for gid, color in chain) + "\n"
      )

  games_mb = os.path.getsize(games_out) / (1024 * 1024)
  chains_mb = os.path.getsize(chains_out) / (1024 * 1024)
  new_lens = sorted(out_len_hist.keys())
  log(
    f"done input_chains={input_chains} kept_chains={len(kept_chains)} "
    f"dropped_short={dropped_short} parse_skip={parse_skip} "
    f"unique_games_out={written} games_mb={games_mb:.2f} chains_mb={chains_mb:.2f}"
  )
  log(f"input chain length hist={dict(sorted(in_len_hist.items()))}")
  log(f"output chain length hist={dict(sorted(out_len_hist.items()))}")
  log(
    f"NOTE: set bt_num_rank_per_batch (and learner_batch_size / nn_rank_size) "
    f"to new chain length {new_lens[0] if len(new_lens) == 1 else new_lens}"
  )


if __name__ == "__main__":
  main()
