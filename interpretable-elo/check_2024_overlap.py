"""Count exact downloaded-game overlap with model corpus subsets."""

import glob
import re
from collections import Counter
from pathlib import Path

import chess.pgn

DATA_DIR = Path("/tmp/blitz-2024-01")
PLAYERS = ("caolinita", "CellblockD", "Fins", "fiodor_nabokov", "getting_there", "TRG80")
HEADER = re.compile(r"RE\[([^]]+)\].*PW\[([^]]+)\]PB\[([^]]+)\]DT\[([^]]+)\]")
MOVE = re.compile(r";[BW]\[([^]]+)\]")


def pgn_key(game):
  result = {"1-0": "1.0", "0-1": "-1.0"}.get(game.headers.get("Result"), "0.0")
  return (
    result,
    game.headers.get("White", "").casefold(),
    game.headers.get("Black", "").casefold(),
    game.headers.get("UTCDate", game.headers.get("Date", "")),
    tuple(move.uci() for move in game.mainline_moves()),
  )


def sgf_key(line):
  header = HEADER.search(line)
  if header is None:
    return None
  result, white, black, date = header.groups()
  return result, white.casefold(), black.casefold(), date, tuple(MOVE.findall(line))


downloaded = {}
owner = {}
for player in PLAYERS:
  with (DATA_DIR / f"{player}-blitz-2024-01.pgn").open(encoding="utf-8", errors="replace") as stream:
    while game := chess.pgn.read_game(stream):
      key = pgn_key(game)
      downloaded[key] = player
      owner[key] = player

print(f"downloaded_unique={len(downloaded)}", flush=True)
for label, paths in (
  ("training", ["/workspace/training_sgf_chess_chain/games.txt"]),
  ("candidate", glob.glob("/workspace/candidate_sgf_chess/*.txt")),
  ("query", glob.glob("/workspace/query_sgf_chess/*.txt")),
):
  hits = Counter()
  lines = 0
  for path in paths:
    with open(path, encoding="utf-8", errors="replace") as stream:
      for line in stream:
        lines += 1
        key = sgf_key(line)
        if key in downloaded:
          hits[owner[key]] += 1
  print(f"{label}: lines={lines} overlaps={sum(hits.values())} by_player={dict(hits)}", flush=True)
