import os
import random
import re
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

min_elo = 1000
max_elo = 2600
interval = 200

# Keep the existing directory naming convention and evaluation quota.
lines_per_file = 50000
evaluation_lines_per_file = 2200

input_dir = "training_sgf"


def get_rank(elo):
    return (elo - min_elo) // interval


def get_game_rank(line):
    wr_match = re.search(r"WR\[(\d+)\]", line)
    br_match = re.search(r"BR\[(\d+)\]", line)
    if not wr_match or not br_match:
        return None

    wr_rating = int(wr_match.group(1))
    br_rating = int(br_match.group(1))
    if wr_rating < min_elo or wr_rating >= max_elo:
        return None

    rank = get_rank(wr_rating)
    return rank if rank == get_rank(br_rating) else None


def process_file(file_name):
    rank_count = (max_elo - min_elo) // interval
    selected_line_numbers = [[] for _ in range(rank_count)]
    seen_per_rank = [0] * rank_count
    input_file = os.path.join(input_dir, file_name)

    for i in range(min_elo, max_elo, interval):
        train_dir = f"rank_{lines_per_file}_{min_elo}_{max_elo}_{interval}interval/train/sgf_{i}_{i + interval}"
        test_dir = f"rank_{lines_per_file}_{min_elo}_{max_elo}_{interval}interval/test_origin/sgf_{i}_{i + interval}"
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

    print(f"------start {file_name} sample {min_elo}~{max_elo}------")

    # Pass 1: choose a bounded-size random evaluation sample per Elo bin.
    with open(input_file, "r", encoding="utf-8") as f_in:
        for line_number, line in enumerate(tqdm(f_in), start=1):
            rank = get_game_rank(line)
            if rank is None:
                continue

            seen_per_rank[rank] += 1
            sample = selected_line_numbers[rank]
            if len(sample) < evaluation_lines_per_file:
                sample.append(line_number)
            else:
                replacement = random.randrange(seen_per_rank[rank])
                if replacement < evaluation_lines_per_file:
                    sample[replacement] = line_number

    for i in range(min_elo, max_elo, interval):
        print(
            f"{file_name} sample {i}~{i + interval}: {seen_per_rank[get_rank(i)]} lines"
        )

    training_outputs = []
    evaluation_outputs = []
    for i in range(min_elo, max_elo, interval):
        train_dir = f"rank_{lines_per_file}_{min_elo}_{max_elo}_{interval}interval/train/sgf_{i}_{i + interval}"
        test_dir = f"rank_{lines_per_file}_{min_elo}_{max_elo}_{interval}interval/test_origin/sgf_{i}_{i + interval}"
        training_outputs.append(open(os.path.join(train_dir, file_name), "w", encoding="utf-8"))
        evaluation_outputs.append(open(os.path.join(test_dir, file_name), "w", encoding="utf-8"))

    selected_line_numbers = [set(numbers) for numbers in selected_line_numbers]
    try:
        # Pass 2: write selected lines to evaluation and all other eligible lines to training.
        with open(input_file, "r", encoding="utf-8") as f_in:
            for line_number, line in enumerate(f_in, start=1):
                rank = get_game_rank(line)
                if rank is None:
                    continue
                if line_number in selected_line_numbers[rank]:
                    evaluation_outputs[rank].write(line)
                else:
                    training_outputs[rank].write(line)
    finally:
        for output in training_outputs + evaluation_outputs:
            output.close()

    print(f"------finish {file_name} sample {min_elo}~{max_elo}------")


with ThreadPoolExecutor() as executor:
    list(executor.map(process_file, os.listdir(input_dir)))
print("complete!")
