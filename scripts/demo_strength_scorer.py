from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
  parser = argparse.ArgumentParser(description="Score one chess SGF with strength_py")
  parser.add_argument("sgf", type=Path)
  parser.add_argument("--config", type=Path, required=True)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--gpu-id", type=int, default=0)
  args = parser.parse_args()

  from build.chess import strength_py

  scorer = strength_py.StrengthScorer(str(args.config), str(args.checkpoint), args.gpu_id)
  print(scorer.score_sgf(args.sgf.read_text(encoding="utf-8")))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
