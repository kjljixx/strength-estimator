#!/usr/bin/env python3
"""Build win-chain ordinal training data from convert.txt games.

Streams training_sgf/ without loading the full corpus into RAM. Discovers
paths A beat B beat ... and writes a deduped game pool plus chain index.
Each chain is ordered weakest -> strongest for BT training. Training games
are non-edge games of each player in the chain.
"""

from __future__ import annotations

import argparse
import os
import random
import re
import sys
from typing import Dict, List, Optional, Tuple

PW_RE = re.compile(r"PW\[([^\]]+)\]")
PB_RE = re.compile(r"PB\[([^\]]+)\]")
RE_RE = re.compile(r"RE\[([^\]]+)\]")

# (file_id, byte_offset, color 'W'|'B')
GameRef = Tuple[int, int, str]
# (loser_id, file_id, byte_offset)
EdgeRef = Tuple[int, int, int]


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


def pass1_stream(
  files: List[str],
  max_games_per_player: int,
  max_out_edges: int,
  max_input_games: int,
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


def sample_chains(
  player_games: List[List[GameRef]],
  out_edges: List[List[EdgeRef]],
  chain_length: int,
  max_chains: int,
  max_attempts: int,
) -> List[List[GameRef]]:
  """Each chain: list of GameRef weak->strong."""
  n = len(player_games)
  starters = [i for i in range(n) if out_edges[i]]
  if not starters:
    print("WARN: no players with outgoing win edges", flush=True)
    return []

  chains: List[List[GameRef]] = []
  attempts = 0
  while len(chains) < max_chains and attempts < max_attempts:
    attempts += 1
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
    if len(chains) % 1000 == 0:
      log_stats(f"sampled chains={len(chains)} attempts={attempts}")

  log_stats(f"finished sampling chains={len(chains)} attempts={attempts}")
  if len(chains) < max_chains:
    print(
      f"WARN: only got {len(chains)}/{max_chains} chains after {attempts} attempts",
      flush=True,
    )
  return chains


def read_line_at(files: List[str], file_id: int, offset: int) -> str:
  with open(files[file_id], "rb") as f:
    f.seek(offset)
    raw = f.readline()
  return raw.decode("utf-8")


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
  args = parser.parse_args()

  if args.chain_length < 2:
    print("ERROR: --chain-length must be >= 2", flush=True)
    sys.exit(1)
  if args.max_attempts <= 0:
    args.max_attempts = max(20 * args.max_chains, 1000)

  random.seed(args.seed)
  files = list_input_files(args.input_dir, args.input_files)

  print("=== sgf_filter_win_chain config ===", flush=True)
  print(f"input_files={len(files)}", flush=True)
  for p in files:
    print(f"  {p}", flush=True)
  print(
    f"output_dir={args.output_dir} chain_length={args.chain_length} "
    f"max_chains={args.max_chains} max_games_per_player={args.max_games_per_player} "
    f"max_out_edges={args.max_out_edges} max_input_games={args.max_input_games} "
    f"max_attempts={args.max_attempts} seed={args.seed}",
    flush=True,
  )
  log_stats("start")

  id_to_name, player_games, out_edges, p1_stats = pass1_stream(
    files,
    args.max_games_per_player,
    args.max_out_edges,
    args.max_input_games,
  )
  log_stats(
    f"pass1 done lines={p1_stats['lines']} decisive={p1_stats['decisive']} "
    f"draws={p1_stats['draws']} parse_skip={p1_stats['parse_skip']} "
    f"players={len(id_to_name)}"
  )

  chains = sample_chains(
    player_games,
    out_edges,
    args.chain_length,
    args.max_chains,
    args.max_attempts,
  )
  if not chains:
    print("ERROR: no chains sampled", flush=True)
    sys.exit(1)

  out_stats = write_outputs(files, chains, args.output_dir)
  log_stats(
    f"done unique_games={out_stats['unique_games']} chains={out_stats['num_chains']} "
    f"games_mb={out_stats['games_mb']:.2f} chains_mb={out_stats['chains_mb']:.2f}"
  )


if __name__ == "__main__":
  main()
