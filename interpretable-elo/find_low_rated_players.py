#!/usr/bin/env python3
"""Download rapid games of low-rated Lichess players found as opponents in local PGNs."""

from __future__ import annotations

import argparse
import os
import random
import re
import sys
import time
from pathlib import Path

import requests

OPPONENT_PATTERN = re.compile(r'^\[(White|Black) "([^"]+)"\]', re.MULTILINE)
API = "https://lichess.org/api"


def opponent_pool(pgn_dir: Path, exclude: set[str]) -> list[str]:
  names = set()
  for path in pgn_dir.rglob("*.pgn"):
    names.update(name for _, name in OPPONENT_PATTERN.findall(path.read_text(encoding="utf-8", errors="replace")))
  return sorted(names - exclude)


def current_rapid_ratings(names: list[str]) -> dict[str, dict]:
  ratings = {}
  for start in range(0, len(names), 300):
    response = requests.post(f"{API}/users", data=",".join(names[start:start + 300]), timeout=60)
    response.raise_for_status()
    for user in response.json():
      rapid = user.get("perfs", {}).get("rapid")
      if rapid and not user.get("closed") and not user.get("tosViolation"):
        ratings[user["username"]] = rapid
    time.sleep(1)
  return ratings


def download_games(player: str, count: int, output: Path) -> int:
  response = requests.get(f"{API}/games/user/{player}", stream=True, timeout=120,
                          params={"max": count, "perfType": "rapid", "rated": "true", "clocks": "false",
                                  "evals": "false", "opening": "false", "moves": "true"},
                          headers={"Accept": "application/x-chess-pgn", **({"Authorization": f"Bearer {os.environ['LICHESS_TOKEN']}"} if "LICHESS_TOKEN" in os.environ else {})})
  if response.status_code == 429:
    print("Rate limited; sleeping 120s", file=sys.stderr)
    time.sleep(120)
    return download_games(player, count, output)
  response.raise_for_status()
  output.write_text(response.text, encoding="utf-8")
  return response.text.count('[Event "')


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--pgn-dir", type=Path, default=Path("data"))
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--players", type=int, required=True)
  parser.add_argument("--games", type=int, default=500)
  parser.add_argument("--max-rating", type=int, default=1500)
  parser.add_argument("--seed", type=int, default=0)
  args = parser.parse_args()
  print(f"Config: pgn_dir={args.pgn_dir}, output_dir={args.output_dir}, players={args.players}, "
        f"games={args.games}, max_rating={args.max_rating}, seed={args.seed}")

  args.output_dir.mkdir(parents=True, exist_ok=True)
  done = {path.name.split("-rapid-")[0] for path in args.output_dir.glob("*.pgn")}
  pool = opponent_pool(args.pgn_dir, done)
  random.Random(args.seed).shuffle(pool)
  print(f"Opponent pool: {len(pool)} names, already downloaded: {len(done)}")

  saved = len(done)
  for start in range(0, len(pool), 300):
    if saved >= args.players:
      break
    ratings = current_rapid_ratings(pool[start:start + 300])
    eligible = [name for name, rapid in ratings.items()
                if rapid["rating"] < args.max_rating and not rapid.get("prov") and rapid["games"] >= args.games]
    print(f"Batch {start // 300}: {len(ratings)} rated, {len(eligible)} eligible")
    for name in eligible:
      if saved >= args.players:
        break
      try:
        count = download_games(name, args.games, args.output_dir / f"{name}-rapid-{args.games}.pgn")
      except requests.HTTPError as error:
        print(f"  {name}: skipped ({error})", file=sys.stderr)
        continue
      saved += 1
      print(f"  {name}: rapid={ratings[name]['rating']} downloaded={count} ({saved}/{args.players})")
      time.sleep(1)
  print(f"Done: {saved} players in {args.output_dir}")
  return 0 if saved >= args.players else 1


if __name__ == "__main__":
  raise SystemExit(main())
