import argparse
import os
import random


def process_file(in_file, cand, test):
  num_lines = cand + test
  with open(in_file, "r") as file:
    lines = file.readlines()
  selected_lines = random.sample(lines, min(num_lines, len(lines)))
  cand_lines = selected_lines[:cand]
  test_lines = selected_lines[cand:]
  for split, split_lines in (("cand", cand_lines), ("test", test_lines)):
    out_file = in_file.replace("test_origin", split)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as file:
      file.writelines(split_lines)
  print(f"{in_file} finished: cand={len(cand_lines)} test={len(test_lines)}")


def process_folder(folder_path, cand, test):
  for root, _, files in os.walk(os.path.join(folder_path, "test_origin")):
    if not os.path.basename(root).startswith("sgf_"):
      continue
    for file in files:
      if file.endswith(".txt"):
        process_file(os.path.join(root, file), cand, test)


def main():
  parser = argparse.ArgumentParser(description="Split test_origin into candidate and query sets.")
  parser.add_argument("--root", default="rank_50000_1000_2600_200interval", help="directory containing test_origin/")
  parser.add_argument("--cand", type=int, default=200)
  parser.add_argument("--test", type=int, default=2000)
  parser.add_argument("--seed", type=int, default=0)
  args = parser.parse_args()
  print(f"config: {vars(args)}")
  random.seed(args.seed)
  process_folder(args.root, args.cand, args.test)


if __name__ == "__main__":
  main()
