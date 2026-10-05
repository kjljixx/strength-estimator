import argparse
import hashlib
import os
import random
import re
import shutil
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "interpretable-elo"))
from phase_rules import boards_after_moves, phase_boundaries, phase_ranges

WHITE_RATING = re.compile(r"WR\[(\d+)\]")
BLACK_RATING = re.compile(r"BR\[(\d+)\]")
FIRST_MOVE = re.compile(r";[BW]\[")
UCI_MOVE = re.compile(r";[BW]\[([a-h][1-8][a-h][1-8][qrbn]?)\]")


def build_parser():
  parser = argparse.ArgumentParser(description="Split SGF games into Elo-binned training and evaluation sets.")
  parser.add_argument("--input", default="training_sgf", help="SGF file, or directory of SGF files")
  parser.add_argument("--output-root", default="rank_50000_1000_2600_200interval")
  parser.add_argument("--min-elo", type=int, default=1000)
  parser.add_argument("--max-elo", type=int, default=2600)
  parser.add_argument("--elo-interval", type=int, default=200)
  parser.add_argument("--compare-interval", type=int, default=200, help="interval used only to report how many games narrower bins exclude")
  parser.add_argument("--eval-per-bin", type=int, default=2200)
  parser.add_argument("--max-train-per-bin", type=int, default=0, help="randomly keep at most this many training games per bin (0 keeps all)")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--phase-examples", action="store_true", help="add PM[]/PE[] midgame/endgame start move indices (-1 if absent) to training games")
  parser.add_argument("--workers", type=int, default=os.cpu_count())
  parser.add_argument("--held-out-dir", action="append", default=[], help="directory of held-out SGFs to exclude entirely")
  parser.add_argument("--stats-only", action="store_true", help="only count games per bin, write nothing")
  return parser


def fingerprint(line):
  return hashlib.blake2b(line.strip().encode("utf-8"), digest_size=8).digest()


def load_held_out(directories):
  held_out = set()
  for directory in directories:
    files = sorted(Path(directory).glob("*.txt"))
    if not files:
      sys.exit(f"ERROR: no held-out .txt files in {directory}")
    for path in files:
      with open(path, "r", encoding="utf-8") as stream:
        held_out.update(fingerprint(line) for line in stream if line.strip())
    print(f"held-out {directory}: {len(held_out)} games so far")
  return held_out


def elo_bin(rating, args, interval):
  if rating < args.min_elo or rating >= args.max_elo:
    return None
  return (rating - args.min_elo) // interval


def classify(line, args, held_out):
  white = WHITE_RATING.search(line)
  black = BLACK_RATING.search(line)
  if not white or not black:
    return None, "missing_rating"
  white_bin = elo_bin(int(white.group(1)), args, args.elo_interval)
  black_bin = elo_bin(int(black.group(1)), args, args.elo_interval)
  if white_bin is None or black_bin is None:
    return None, "rating_out_of_range"
  if white_bin != black_bin:
    return None, "players_in_different_bins"
  if held_out and fingerprint(line) in held_out:
    return None, "held_out_game"
  return white_bin, None


def compare_interval_keeps(line, args):
  white = WHITE_RATING.search(line)
  black = BLACK_RATING.search(line)
  if not white or not black:
    return False
  white_bin = elo_bin(int(white.group(1)), args, args.compare_interval)
  return white_bin is not None and white_bin == elo_bin(int(black.group(1)), args, args.compare_interval)


def add_phase_tags(line):
  moves = UCI_MOVE.findall(line)
  if not moves:
    return None, "no_uci_moves", ()
  try:
    middle, end = phase_boundaries(boards_after_moves(moves))
  except ValueError:
    return None, "illegal_move_replay", ()
  tags = f"PM[{-1 if middle is None else middle}]PE[{-1 if end is None else end}]"
  first_move = FIRST_MOVE.search(line).start()
  return line[:first_move] + tags + line[first_move:], None, tuple(phase_ranges(len(moves), middle, end))


def process_task(task):
  rank, is_evaluation, line, phase_examples = task
  if is_evaluation or not phase_examples:
    return rank, is_evaluation, line, None, ()
  annotated, reason, phases = add_phase_tags(line)
  return rank, is_evaluation, annotated, reason, phases


def reservoir_add(sample, line_number, seen_count, capacity, rng):
  if len(sample) < capacity:
    sample.append(line_number)
    return
  replacement = rng.randrange(seen_count)
  if replacement < capacity:
    sample[replacement] = line_number


def scan_input(input_file, args, held_out, rng):
  rank_count = (args.max_elo - args.min_elo) // args.elo_interval
  stats = Counter()
  seen_per_rank = [0] * rank_count
  evaluation_samples = [[] for _ in range(rank_count)]
  training_samples = [[] for _ in range(rank_count)]
  training_capacity = args.max_train_per_bin + args.eval_per_bin
  with open(input_file, "r", encoding="utf-8") as stream:
    for line_number, line in enumerate(tqdm(stream, desc=f"scan {input_file.name}"), start=1):
      stats["games"] += 1
      rank, reason = classify(line, args, held_out)
      if rank is None:
        stats[reason] += 1
        continue
      seen_per_rank[rank] += 1
      reservoir_add(evaluation_samples[rank], line_number, seen_per_rank[rank], args.eval_per_bin, rng)
      if args.max_train_per_bin:
        reservoir_add(training_samples[rank], line_number, seen_per_rank[rank], training_capacity, rng)
  with open(input_file, "r", encoding="utf-8") as stream:
    stats["kept_at_compare_interval"] = sum(compare_interval_keeps(line, args) for line in tqdm(stream, desc="compare interval"))
  evaluation_lines = [set(sample) for sample in evaluation_samples]
  training_lines = None
  if args.max_train_per_bin:
    training_lines = []
    for rank in range(rank_count):
      candidates = [number for number in training_samples[rank] if number not in evaluation_lines[rank]]
      rng.shuffle(candidates)
      training_lines.append(set(candidates[:args.max_train_per_bin]))
  return stats, seen_per_rank, evaluation_lines, training_lines


def bin_directory(args, split, rank):
  low = args.min_elo + rank * args.elo_interval
  return Path(args.output_root) / split / f"sgf_{low}_{low + args.elo_interval}"


def route_tasks(input_file, args, held_out, evaluation_lines, training_lines):
  with open(input_file, "r", encoding="utf-8") as stream:
    for line_number, line in enumerate(stream, start=1):
      rank, _ = classify(line, args, held_out)
      if rank is None:
        continue
      is_evaluation = line_number in evaluation_lines[rank]
      if is_evaluation or training_lines is None or line_number in training_lines[rank]:
        yield rank, is_evaluation, line, args.phase_examples


def write_outputs(input_file, args, held_out, evaluation_lines, training_lines):
  rank_count = len(evaluation_lines)
  outputs = {}
  for rank in range(rank_count):
    for split in ("train", "test_origin"):
      directory = bin_directory(args, split, rank)
      directory.mkdir(parents=True, exist_ok=True)
      outputs[(rank, split == "test_origin")] = open(directory / input_file.name, "w", encoding="utf-8")

  written = Counter()
  skipped = Counter()
  phase_games = [Counter() for _ in range(rank_count)]
  tasks = route_tasks(input_file, args, held_out, evaluation_lines, training_lines)
  pool = Pool(args.workers) if args.phase_examples and args.workers > 1 else None
  results = pool.imap(process_task, tasks, chunksize=1000) if pool else map(process_task, tasks)
  try:
    for rank, is_evaluation, line, reason, phases in tqdm(results, desc=f"write {input_file.name}"):
      if reason is not None:
        skipped[reason] += 1
        continue
      outputs[(rank, is_evaluation)].write(line)
      written[(rank, is_evaluation)] += 1
      phase_games[rank].update(phases)
  finally:
    if pool:
      pool.close()
      pool.join()
    for output in outputs.values():
      output.close()
  return written, skipped, phase_games


def report(stats, seen_per_rank, args):
  print(f"games={stats['games']}")
  for reason in ("missing_rating", "rating_out_of_range", "players_in_different_bins", "held_out_game"):
    print(f"excluded {reason}={stats[reason]}")
  kept = sum(seen_per_rank)
  print(f"kept at interval {args.elo_interval}={kept}")
  print(f"kept at interval {args.compare_interval}={stats['kept_at_compare_interval']} (held-out games not removed)")
  for rank, count in enumerate(seen_per_rank):
    low = args.min_elo + rank * args.elo_interval
    print(f"bin {low}-{low + args.elo_interval - 1}: eligible games={count}")


def check_free_disk(input_file, stats, seen_per_rank, args):
  games = sum(seen_per_rank)
  kept = sum(min(count, args.max_train_per_bin) if args.max_train_per_bin else count for count in seen_per_rank)
  kept += min(games, args.eval_per_bin * len(seen_per_rank))
  average_bytes = input_file.stat().st_size / stats["games"]
  needed = 2 * kept * average_bytes
  Path(args.output_root).mkdir(parents=True, exist_ok=True)
  free = shutil.disk_usage(args.output_root).free
  print(f"estimated output (split + merged copies)={needed / 1e9:.1f} GB, free disk={free / 1e9:.1f} GB")
  if needed * 1.2 > free:
    sys.exit("ERROR: not enough free disk; lower --max-train-per-bin or free space")


def process_file(input_file, args, held_out):
  rng = random.Random(f"{args.seed}:{input_file.name}")
  print(f"------start {input_file.name}------")
  stats, seen_per_rank, evaluation_lines, training_lines = scan_input(input_file, args, held_out, rng)
  report(stats, seen_per_rank, args)
  empty_bins = [rank for rank, count in enumerate(seen_per_rank) if count <= args.eval_per_bin]
  if empty_bins:
    print(f"WARNING: bins with eligible games <= eval-per-bin (no training games left): {empty_bins}")
  if args.stats_only:
    return
  check_free_disk(input_file, stats, seen_per_rank, args)
  written, skipped, phase_games = write_outputs(input_file, args, held_out, evaluation_lines, training_lines)
  for rank, counter in enumerate(phase_games):
    print(f"bin {rank}: train={written[(rank, False)]} eval={written[(rank, True)]} games_with_phase={dict(counter)}")
  print(f"skipped while annotating={dict(skipped)}")
  print(f"------finish {input_file.name}------")


def main():
  args = build_parser().parse_args()
  print(f"config: {vars(args)}", flush=True)
  if (args.max_elo - args.min_elo) % args.elo_interval:
    sys.exit("ERROR: elo range must be a multiple of elo-interval")
  if not args.held_out_dir:
    print("WARNING: no held-out directories given; training may include evaluation games")
  held_out = load_held_out(args.held_out_dir)
  source = Path(args.input)
  input_files = sorted(source.iterdir()) if source.is_dir() else [source]
  for input_file in input_files:
    process_file(input_file, args, held_out)
  print("complete!")


if __name__ == "__main__":
  main()
