"""Select original PGNs matching games emitted by the legacy converter."""

import re
from collections import Counter
from pathlib import Path

import chess.pgn

LEGACY = Path("/tmp/legacy-fiodor/converted.txt")
ORIGINAL = Path("/tmp/blitz-2024-01/fiodor_nabokov-blitz-2024-01.pgn")
OUTPUT = Path("/tmp/legacy-fiodor/matched-original.pgn")


def legacy_key(line):
  def tag(name):
    match = re.search(rf"{name}\[([^]]*)\]", line)
    return match.group(1) if match else ""
  return tag("PW").casefold(), tag("PB").casefold(), tag("DT"), tag("RE")


keys = Counter(legacy_key(line) for line in LEGACY.open(encoding="utf-8", errors="replace"))
matched = 0
with ORIGINAL.open(encoding="utf-8", errors="replace") as source, OUTPUT.open("w", encoding="utf-8") as target:
  exporter = chess.pgn.FileExporter(target)
  while game := chess.pgn.read_game(source):
    result = {"1-0": "1.0", "0-1": "-1.0"}.get(game.headers.get("Result"), "0.0")
    key = (
      game.headers.get("White", "").casefold(),
      game.headers.get("Black", "").casefold(),
      game.headers.get("UTCDate", game.headers.get("Date", "")),
      result,
    )
    if keys[key] > 0:
      game.accept(exporter)
      keys[key] -= 1
      matched += 1
print(f"matched={matched} unmatched_legacy={sum(keys.values())}")
