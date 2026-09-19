"""Compare legacy and python-chess move encodings for identical games."""

import re
import sys
from collections import Counter
from pathlib import Path

import chess.pgn

sys.path.insert(0, "/tmp/interpretable-overall")
from analyze import game_to_sgf

MOVE = re.compile(r";[BW]\[([^]]+)\]")
legacy_lines = list(Path("/tmp/legacy-fiodor/converted.txt").open(encoding="utf-8"))
games = []
with Path("/tmp/legacy-fiodor/matched-original.pgn").open(encoding="utf-8") as stream:
  while game := chess.pgn.read_game(stream):
    games.append(game)

identical = 0
first_differences = Counter()
examples = []
for index, (line, game) in enumerate(zip(legacy_lines, games), start=1):
  legacy = MOVE.findall(line)
  current = MOVE.findall(game_to_sgf(game)[0])
  if legacy == current:
    identical += 1
    continue
  position = next((i for i, pair in enumerate(zip(legacy, current), start=1) if pair[0] != pair[1]), None)
  if position is None:
    position = min(len(legacy), len(current)) + 1
  old = legacy[position - 1] if position <= len(legacy) else "<end>"
  new = current[position - 1] if position <= len(current) else "<end>"
  first_differences[(old, new)] += 1
  if len(examples) < 10:
    examples.append((index, position, old, new, len(legacy), len(current)))
print(f"games={len(games)} identical={identical} different={len(games) - identical}")
print(f"top_first_differences={first_differences.most_common(15)}")
print(f"examples={examples}")
