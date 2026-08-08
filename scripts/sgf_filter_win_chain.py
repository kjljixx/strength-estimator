#!/usr/bin/env python3
"""Build win-chain ordinal training data from convert.txt games.

Streams training_sgf/ without loading the full corpus into RAM. Discovers
paths A beat B beat ... and writes a deduped game pool plus chain index.
Each chain is ordered weakest -> strongest for BT training. Training games
are non-edge games of each player in the chain.

Also writes plot_data.json (no plotting) for scripts/plot_win_chain.py,
flushed after pass1 and periodically during chain sampling.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter
from typing import Dict, List, Optional, Tuple

PW_RE = re.compile(r"PW\[([^\]]+)\]")
PB_RE = re.compile(r"PB\[([^\]]+)\]")
RE_RE = re.compile(r"RE\[([^\]]+)\]")
WR_RE = re.compile(r"WR\[(\d+)\]")
BR_RE = re.compile(r"BR\[(\d+)\]")

# (file_id, byte_offset, color 'W'|'B')
GameRef = Tuple[int, int, str]
# (loser_id, file_id, byte_offset)
EdgeRef = Tuple[int, int, int]

PLAYER_COUNT_FLUSH_TOP = 500
ABS_ELO_HIST_LO = 0.0
ABS_ELO_HIST_HI = 2000.0
ABS_ELO_HIST_BINS = 80
EDGE_ELO_HIST_LO = -1500.0
EDGE_ELO_HIST_HI = 1500.0
EDGE_ELO_HIST_BINS = 120


def peak_rss_mb() -> Optional[float]:
  try:
    import psutil

    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
  except Exception:
    try:
      import resource

      # Linux: ru_maxrss is KB; macOS: bytes
      rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
      if sys.platform == "darwin":
        return rss / (1024 * 1024)
      return rss / 1024.0
    except Exception:
      return None


def log_stats(msg: str) -> None:
  rss = peak_rss_mb()
  if rss is None:
    print(msg, flush=True)
  else:
    print(f"{msg} | rss={rss:.1f}MB", flush=True)


def reservoir_push(buf: list, item, cap: int, seen: int) -> None:
  """1-based seen count of items considered for this buffer."""
  if cap <= 0:
    return
  if len(buf) < cap:
    buf.append(item)
    return
  j = random.randint(0, seen - 1)
  if j < cap:
    buf[j] = item


def intern_player(name: str, name_to_id: Dict[str, int], id_to_name: List[str]) -> int:
  pid = name_to_id.get(name)
  if pid is not None:
    return pid
  pid = len(id_to_name)
  name_to_id[name] = pid
  id_to_name.append(name)
  return pid


def parse_result(line: str) -> Optional[float]:
  m = RE_RE.search(line)
  if not m:
    return None
  try:
    return float(m.group(1))
  except ValueError:
    print(f"WARN: bad RE tag: {m.group(1)!r}", flush=True)
    return None


def parse_ratings(line: str) -> Optional[Tuple[int, int]]:
  wr_m, br_m = WR_RE.search(line), BR_RE.search(line)
  if not wr_m or not br_m:
    return None
  try:
    return int(wr_m.group(1)), int(br_m.group(1))
  except ValueError:
    return None


def list_input_files(input_dir: str, input_files: Optional[List[str]]) -> List[str]:
  if input_files:
    paths = []
    for p in input_files:
      if not os.path.isfile(p):
        raise FileNotFoundError(p)
      paths.append(p)
    return paths
  if not os.path.isdir(input_dir):
    raise FileNotFoundError(f"input dir not found: {input_dir}")
  paths = sorted(
    os.path.join(input_dir, name)
    for name in os.listdir(input_dir)
    if os.path.isfile(os.path.join(input_dir, name))
  )
  if not paths:
    raise FileNotFoundError(f"no files in {input_dir}")
  return paths


class Moments:
  __slots__ = ("n", "sum", "sum_sq")

  def __init__(self) -> None:
    self.n = 0
    self.sum = 0.0
    self.sum_sq = 0.0

  def add(self, x: float) -> None:
    self.n += 1
    self.sum += x
    self.sum_sq += x * x

  def to_dict(self) -> dict:
    return {"n": self.n, "sum": self.sum, "sum_sq": self.sum_sq}


class FixedHistogram:
  def __init__(self, lo: float, hi: float, n_bins: int) -> None:
    if hi <= lo or n_bins <= 0:
      raise ValueError("invalid histogram bounds")
    self.lo = lo
    self.hi = hi
    self.n_bins = n_bins
    self.counts = [0] * n_bins
    self.underflow = 0
    self.overflow = 0
    self.moments = Moments()

  def add(self, x: float) -> None:
    self.moments.add(x)
    if x < self.lo:
      self.underflow += 1
      return
    if x >= self.hi:
      self.overflow += 1
      return
    width = self.hi - self.lo
    idx = int((x - self.lo) / width * self.n_bins)
    if idx >= self.n_bins:
      idx = self.n_bins - 1
    self.counts[idx] += 1

  def to_dict(self) -> dict:
    return {
      "lo": self.lo,
      "hi": self.hi,
      "n_bins": self.n_bins,
      "counts": self.counts,
      "underflow": self.underflow,
      "overflow": self.overflow,
      "moments": self.moments.to_dict(),
    }


class PlotDataCollector:
  """Numeric aggregates for plot_win_chain.py (no matplotlib)."""

  def __init__(
    self,
    path: str,
    chain_length: int,
    config: dict,
  ) -> None:
    self.path = path
    self.chain_length = chain_length
    self.config = config
    self.phase = "init"
    self.abs_elo_diff_all = FixedHistogram(ABS_ELO_HIST_LO, ABS_ELO_HIST_HI, ABS_ELO_HIST_BINS)
    self.chain_edge_elo_diffs = FixedHistogram(EDGE_ELO_HIST_LO, EDGE_ELO_HIST_HI, EDGE_ELO_HIST_BINS)
    self.elo_rel_by_link = [Moments() for _ in range(chain_length)]
    self.color_by_slot = [[0, 0] for _ in range(chain_length)]  # [W, B]
    self.player_counts: Counter = Counter()
    self.sampling = {
      "chains": 0,
      "attempts": 0,
      "max_chains": config.get("max_chains", 0),
      "max_attempts": config.get("max_attempts", 0),
    }
    self.pass1 = {}
    self.edge_elo_skip = 0

  def set_phase(self, phase: str) -> None:
    self.phase = phase

  def set_pass1(self, stats: dict, n_players: int) -> None:
    self.pass1 = {
      "lines": stats.get("lines", 0),
      "decisive": stats.get("decisive", 0),
      "draws": stats.get("draws", 0),
      "parse_skip": stats.get("parse_skip", 0),
      "rating_skip": stats.get("rating_skip", 0),
      "players": n_players,
    }

  def add_abs_elo_diff(self, abs_diff: float) -> None:
    self.abs_elo_diff_all.add(abs_diff)

  def record_accepted_chain(
    self,
    id_to_name: List[str],
    weak_to_strong: List[int],
    train_refs: List[GameRef],
    elos: List[float],
    edge_diffs: List[float],
  ) -> None:
    for pid in weak_to_strong:
      self.player_counts[id_to_name[pid]] += 1
    for slot, (_fid, _off, color) in enumerate(train_refs):
      if color == "W":
        self.color_by_slot[slot][0] += 1
      else:
        self.color_by_slot[slot][1] += 1
    if elos:
      base = elos[0]
      for k, elo in enumerate(elos):
        self.elo_rel_by_link[k].add(elo - base)
    for d in edge_diffs:
      self.chain_edge_elo_diffs.add(d)

  def flush(self) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(self.path)) or ".", exist_ok=True)
    top = self.player_counts.most_common(PLAYER_COUNT_FLUSH_TOP)
    payload = {
      "meta": {
        "phase": self.phase,
        "chain_length": self.chain_length,
        "config": self.config,
      },
      "pass1": self.pass1,
      "sampling": dict(self.sampling),
      "player_counts": {
        "top": [[name, int(c)] for name, c in top],
        "unique_players_in_chains": len(self.player_counts),
        "total_slot_appearances": int(sum(self.player_counts.values())),
      },
      "color_by_slot": self.color_by_slot,
      "elo_rel_by_link": [m.to_dict() for m in self.elo_rel_by_link],
      "chain_edge_elo_diffs": self.chain_edge_elo_diffs.to_dict(),
      "abs_elo_diff_all": self.abs_elo_diff_all.to_dict(),
      "edge_elo_skip": self.edge_elo_skip,
    }
    tmp = self.path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
      json.dump(payload, f, separators=(",", ":"))
    os.replace(tmp, self.path)
    log_stats(f"wrote plot_data phase={self.phase} path={self.path}")


def pass1_stream(
  files: List[str],
  max_games_per_player: int,
  max_out_edges: int,
  max_input_games: int,
  plot: Optional[PlotDataCollector] = None,
) -> Tuple[List[str], List[List[GameRef]], List[List[EdgeRef]], dict]:
  name_to_id: Dict[str, int] = {}
  id_to_name: List[str] = []
  player_games: List[List[GameRef]] = []
  player_game_seen: List[int] = []
  out_edges: List[List[EdgeRef]] = []
  out_edge_seen: List[int] = []

  stats = {
    "lines": 0,
    "decisive": 0,
    "draws": 0,
    "parse_skip": 0,
    "rating_skip": 0,
    "files": len(files),
  }

  def ensure_player(pid: int) -> None:
    while len(player_games) <= pid:
      player_games.append([])
      player_game_seen.append(0)
      out_edges.append([])
      out_edge_seen.append(0)

  for file_id, path in enumerate(files):
    log_stats(f"pass1 reading file {file_id + 1}/{len(files)}: {path}")
    with open(path, "rb") as f:
      while True:
        if max_input_games > 0 and stats["lines"] >= max_input_games:
          log_stats(f"pass1 early stop at max_input_games={max_input_games}")
          return id_to_name, player_games, out_edges, stats

        offset = f.tell()
        raw = f.readline()
        if not raw:
          break
        stats["lines"] += 1
        if stats["lines"] % 1_000_000 == 0:
          log_stats(
            f"pass1 lines={stats['lines']} players={len(id_to_name)} "
            f"decisive={stats['decisive']}"
          )

        try:
          line = raw.decode("utf-8")
        except UnicodeDecodeError:
          stats["parse_skip"] += 1
          continue

        pw_m, pb_m = PW_RE.search(line), PB_RE.search(line)
        result = parse_result(line)
        if not pw_m or not pb_m or result is None:
          stats["parse_skip"] += 1
          continue
        if result not in (1.0, -1.0, 0.0):
          stats["parse_skip"] += 1
          continue

        ratings = parse_ratings(line)
        if ratings is None:
          stats["rating_skip"] += 1
        elif plot is not None:
          wr, br = ratings
          plot.add_abs_elo_diff(abs(wr - br))

        white, black = pw_m.group(1), pb_m.group(1)
        wid = intern_player(white, name_to_id, id_to_name)
        bid = intern_player(black, name_to_id, id_to_name)
        ensure_player(max(wid, bid))

        player_game_seen[wid] += 1
        reservoir_push(player_games[wid], (file_id, offset, "W"), max_games_per_player, player_game_seen[wid])
        player_game_seen[bid] += 1
        reservoir_push(player_games[bid], (file_id, offset, "B"), max_games_per_player, player_game_seen[bid])

        if result == 0.0:
          stats["draws"] += 1
          continue

        stats["decisive"] += 1
        if result == 1.0:
          winner, loser = wid, bid
        else:
          winner, loser = bid, wid
        out_edge_seen[winner] += 1
        reservoir_push(out_edges[winner], (loser, file_id, offset), max_out_edges, out_edge_seen[winner])

  return id_to_name, player_games, out_edges, stats


def sample_path(
  start: int,
  out_edges: List[List[EdgeRef]],
  chain_length: int,
) -> Optional[Tuple[List[int], List[Tuple[int, int]]]]:
  """Return (players strong->weak, edge_game refs) or None."""
  if chain_length < 2:
    return None
  players = [start]
  edge_refs: List[Tuple[int, int]] = []
  used = {start}
  cur = start
  for _ in range(chain_length - 1):
    candidates = [e for e in out_edges[cur] if e[0] not in used]
    if not candidates:
      return None
    loser, file_id, offset = random.choice(candidates)
    players.append(loser)
    edge_refs.append((file_id, offset))
    used.add(loser)
    cur = loser
  return players, edge_refs


def pick_train_game(
  pid: int,
  player_games: List[List[GameRef]],
  forbidden: set,
) -> Optional[GameRef]:
  candidates = [g for g in player_games[pid] if (g[0], g[1]) not in forbidden]
  if not candidates:
    return None
  return random.choice(candidates)


def read_line_at(files: List[str], file_id: int, offset: int) -> str:
  with open(files[file_id], "rb") as f:
    f.seek(offset)
    raw = f.readline()
  return raw.decode("utf-8")


def edge_src_dest_elo(
  line: str,
  winner_id: int,
  loser_id: int,
  id_to_name: List[str],
) -> Optional[Tuple[float, float]]:
  """Return (src_elo, dest_elo) = (loser rating, winner rating) on this game."""
  ratings = parse_ratings(line)
  pw_m, pb_m = PW_RE.search(line), PB_RE.search(line)
  if ratings is None or not pw_m or not pb_m:
    return None
  wr, br = ratings
  white, black = pw_m.group(1), pb_m.group(1)
  wname, lname = id_to_name[winner_id], id_to_name[loser_id]
  if white == wname and black == lname:
    return float(br), float(wr)
  if white == lname and black == wname:
    return float(wr), float(br)
  return None


def chain_edge_elos(
  files: List[str],
  id_to_name: List[str],
  weak_to_strong: List[int],
  edge_refs_strong_to_weak: List[Tuple[int, int]],
) -> Optional[Tuple[List[float], List[float]]]:
  """Build per-slot elos and weak->strong edge diffs from path edge games."""
  k = len(weak_to_strong)
  if k < 2 or len(edge_refs_strong_to_weak) != k - 1:
    return None
  elos: List[Optional[float]] = [None] * k
  edge_diffs: List[float] = []
  for j in range(k - 1):
    loser = weak_to_strong[j]
    winner = weak_to_strong[j + 1]
    edge_i = (k - 2) - j
    file_id, offset = edge_refs_strong_to_weak[edge_i]
    line = read_line_at(files, file_id, offset)
    pair = edge_src_dest_elo(line, winner, loser, id_to_name)
    if pair is None:
      return None
    src_elo, dest_elo = pair
    if elos[j] is None:
      elos[j] = src_elo
    elos[j + 1] = dest_elo
    edge_diffs.append(dest_elo - src_elo)
  if any(e is None for e in elos):
    return None
  return [float(e) for e in elos], edge_diffs  # type: ignore[misc]


def sample_chains(
  files: List[str],
  id_to_name: List[str],
  player_games: List[List[GameRef]],
  out_edges: List[List[EdgeRef]],
  chain_length: int,
  max_chains: int,
  max_attempts: int,
  plot: Optional[PlotDataCollector] = None,
  plot_flush_every: int = 1000,
) -> List[List[GameRef]]:
  """Each chain: list of GameRef weak->strong."""
  n = len(player_games)
  starters = [i for i in range(n) if out_edges[i]]
  if not starters:
    print("WARN: no players with outgoing win edges", flush=True)
    return []

  chains: List[List[GameRef]] = []
  attempts = 0
  if plot is not None:
    plot.set_phase("sampling")
    plot.sampling["max_chains"] = max_chains
    plot.sampling["max_attempts"] = max_attempts

  while len(chains) < max_chains and attempts < max_attempts:
    attempts += 1
    if plot is not None:
      plot.sampling["attempts"] = attempts
    start = random.choice(starters)
    sampled = sample_path(start, out_edges, chain_length)
    if sampled is None:
      continue
    strong_to_weak, edge_refs = sampled
    forbidden = set(edge_refs)
    # BT order: weak -> strong
    weak_to_strong = list(reversed(strong_to_weak))
    train_refs: List[GameRef] = []
    ok = True
    for pid in weak_to_strong:
      ref = pick_train_game(pid, player_games, forbidden)
      if ref is None:
        ok = False
        break
      train_refs.append(ref)
      forbidden.add((ref[0], ref[1]))
    if not ok:
      continue
    chains.append(train_refs)

    if plot is not None:
      plot.sampling["chains"] = len(chains)
      elo_pack = chain_edge_elos(files, id_to_name, weak_to_strong, edge_refs)
      if elo_pack is None:
        plot.edge_elo_skip += 1
        plot.record_accepted_chain(id_to_name, weak_to_strong, train_refs, [], [])
      else:
        elos, edge_diffs = elo_pack
        plot.record_accepted_chain(id_to_name, weak_to_strong, train_refs, elos, edge_diffs)
      if plot_flush_every > 0 and len(chains) % plot_flush_every == 0:
        plot.flush()

    if len(chains) % 1000 == 0:
      log_stats(f"sampled chains={len(chains)} attempts={attempts}")

  if plot is not None:
    plot.sampling["chains"] = len(chains)
    plot.sampling["attempts"] = attempts
    plot.set_phase("sampling")
    plot.flush()

  log_stats(f"finished sampling chains={len(chains)} attempts={attempts}")
  if len(chains) < max_chains:
    print(
      f"WARN: only got {len(chains)}/{max_chains} chains after {attempts} attempts",
      flush=True,
    )
  return chains


def write_outputs(
  files: List[str],
  chains: List[List[GameRef]],
  output_dir: str,
) -> dict:
  os.makedirs(output_dir, exist_ok=True)
  # Dedup training games
  key_to_idx: Dict[Tuple[int, int], int] = {}
  game_keys: List[Tuple[int, int]] = []
  chain_rows: List[List[Tuple[int, str]]] = []

  for chain in chains:
    row: List[Tuple[int, str]] = []
    for file_id, offset, color in chain:
      key = (file_id, offset)
      idx = key_to_idx.get(key)
      if idx is None:
        idx = len(game_keys)
        key_to_idx[key] = idx
        game_keys.append(key)
      row.append((idx, color))
    chain_rows.append(row)

  games_path = os.path.join(output_dir, "games.txt")
  chains_path = os.path.join(output_dir, "chains.txt")
  log_stats(f"writing {len(game_keys)} unique games -> {games_path}")
  with open(games_path, "w", encoding="utf-8") as out:
    for i, (file_id, offset) in enumerate(game_keys):
      line = read_line_at(files, file_id, offset)
      if not line.endswith("\n"):
        line += "\n"
      out.write(line)
      if (i + 1) % 50_000 == 0:
        log_stats(f"wrote games {i + 1}/{len(game_keys)}")

  log_stats(f"writing {len(chain_rows)} chains -> {chains_path}")
  with open(chains_path, "w", encoding="utf-8") as out:
    for row in chain_rows:
      out.write(" ".join(f"{idx}:{color}" for idx, color in row) + "\n")

  games_bytes = os.path.getsize(games_path)
  chains_bytes = os.path.getsize(chains_path)
  return {
    "unique_games": len(game_keys),
    "num_chains": len(chain_rows),
    "games_mb": games_bytes / (1024 * 1024),
    "chains_mb": chains_bytes / (1024 * 1024),
  }


def main() -> None:
  parser = argparse.ArgumentParser(description="Build win-chain ordinal chess training data")
  parser.add_argument("--input-dir", default="training_sgf", help="Directory of convert.txt files")
  parser.add_argument("--input", action="append", dest="input_files", help="Explicit input file (repeatable)")
  parser.add_argument("--output-dir", default="training_sgf_chess_chain")
  parser.add_argument("--chain-length", type=int, default=8)
  parser.add_argument("--max-chains", type=int, default=50000)
  parser.add_argument("--max-games-per-player", type=int, default=32)
  parser.add_argument("--max-out-edges", type=int, default=16, help="Cap stored wins per player (RAM)")
  parser.add_argument("--max-input-games", type=int, default=0, help="Stop after N games (0=all)")
  parser.add_argument("--max-attempts", type=int, default=0, help="Chain sample attempts (0=20*max_chains)")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument(
    "--plot-data-path",
    default="",
    help="Path for plot_data.json (default: {output-dir}/plot_data.json)",
  )
  parser.add_argument(
    "--plot-flush-every",
    type=int,
    default=1000,
    help="Rewrite plot_data.json every N accepted chains (0=only after pass1 and end)",
  )
  args = parser.parse_args()

  if args.chain_length < 2:
    print("ERROR: --chain-length must be >= 2", flush=True)
    sys.exit(1)
  if args.max_attempts <= 0:
    args.max_attempts = max(20 * args.max_chains, 1000)

  random.seed(args.seed)
  files = list_input_files(args.input_dir, args.input_files)
  os.makedirs(args.output_dir, exist_ok=True)
  plot_path = args.plot_data_path or os.path.join(args.output_dir, "plot_data.json")
  config = {
    "input_files": len(files),
    "output_dir": args.output_dir,
    "chain_length": args.chain_length,
    "max_chains": args.max_chains,
    "max_games_per_player": args.max_games_per_player,
    "max_out_edges": args.max_out_edges,
    "max_input_games": args.max_input_games,
    "max_attempts": args.max_attempts,
    "seed": args.seed,
    "plot_flush_every": args.plot_flush_every,
  }
  plot = PlotDataCollector(plot_path, args.chain_length, config)

  print("=== sgf_filter_win_chain config ===", flush=True)
  print(f"input_files={len(files)}", flush=True)
  for p in files:
    print(f"  {p}", flush=True)
  print(
    f"output_dir={args.output_dir} chain_length={args.chain_length} "
    f"max_chains={args.max_chains} max_games_per_player={args.max_games_per_player} "
    f"max_out_edges={args.max_out_edges} max_input_games={args.max_input_games} "
    f"max_attempts={args.max_attempts} seed={args.seed} "
    f"plot_data={plot_path} plot_flush_every={args.plot_flush_every}",
    flush=True,
  )
  log_stats("start")

  id_to_name, player_games, out_edges, p1_stats = pass1_stream(
    files,
    args.max_games_per_player,
    args.max_out_edges,
    args.max_input_games,
    plot=plot,
  )
  log_stats(
    f"pass1 done lines={p1_stats['lines']} decisive={p1_stats['decisive']} "
    f"draws={p1_stats['draws']} parse_skip={p1_stats['parse_skip']} "
    f"rating_skip={p1_stats['rating_skip']} players={len(id_to_name)}"
  )
  plot.set_pass1(p1_stats, len(id_to_name))
  plot.set_phase("pass1")
  plot.flush()

  chains = sample_chains(
    files,
    id_to_name,
    player_games,
    out_edges,
    args.chain_length,
    args.max_chains,
    args.max_attempts,
    plot=plot,
    plot_flush_every=args.plot_flush_every,
  )
  if not chains:
    plot.set_phase("done")
    plot.flush()
    print("ERROR: no chains sampled", flush=True)
    sys.exit(1)

  out_stats = write_outputs(files, chains, args.output_dir)
  plot.set_phase("done")
  plot.flush()
  log_stats(
    f"done unique_games={out_stats['unique_games']} chains={out_stats['num_chains']} "
    f"games_mb={out_stats['games_mb']:.2f} chains_mb={out_stats['chains_mb']:.2f}"
  )


if __name__ == "__main__":
  main()
