#!/usr/bin/env python3
"""Stream a user's public Lichess games to a PGN file."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("username")
  parser.add_argument("--max", type=int, default=1000, dest="max_games")
  parser.add_argument("--perf", default="rapid", help="rapid, blitz, classical, bullet, or all")
  parser.add_argument("--output", type=Path)
  args = parser.parse_args()
  if args.max_games < 1:
    parser.error("--max must be positive")

  output = args.output or Path(f"{args.username}-{args.perf}-{args.max_games}.pgn")
  query = urlencode({"max": args.max_games, "rated": "true", "perfType": args.perf})
  url = f"https://lichess.org/api/games/user/{args.username}?{query}"
  request = Request(url, headers={"Accept": "application/x-chess-pgn", "User-Agent": "interpretable-elo/1.0"})

  try:
    with urlopen(request, timeout=60) as response, output.open("wb") as destination:
      shutil.copyfileobj(response, destination)
  except HTTPError as error:
    raise SystemExit(f"Lichess returned HTTP {error.code}: {error.reason}") from error

  print(f"Downloaded public rated games to {output} ({output.stat().st_size:,} bytes)")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
