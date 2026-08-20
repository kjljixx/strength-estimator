import csv
import os
import subprocess
import sys

def parse_line(line):
  tokens = line.strip().split()
  if len(tokens) >= 10 and tokens[0].isdigit():
    game_num = int(tokens[0])
    if 1 <= game_num <= 100:
      return game_num, float(tokens[9])
  return None, None

def run_evaluations(model_dir, start_step=10000, max_step=100000, step_interval=10000):
  results = {}

  for step in range(start_step, max_step + 1, step_interval):
    weight_file = f'{model_dir}/weight_iter_{step}.pt'

    if not os.path.exists(weight_file):
      print(f'\nSkipping step {step}: file {weight_file} does not exist.')
      continue

    cmd = [
      './build/chess/strength_chess',
      '-conf_file', 'cfg/se_chess.cfg',
      '-mode', 'evaluator',
      '-conf_str', f'nn_file_name={weight_file}'
    ]

    print(f"\n{'='*60}\nRunning evaluation for step {step}...\n{'='*60}")

    try:
      proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
      )

      game_acc = {}
      recording = False

      for line in proc.stdout:
        print(line, end='')
        if 'Summarizing accuracy' in line:
          recording = True
          continue

        if recording:
          g_num, acc = parse_line(line)
          if g_num is not None:
            game_acc[g_num] = acc

      proc.wait()

      if proc.returncode != 0:
        print(f'\nProcess crashed with exit code {proc.returncode}')
        continue

      if game_acc:
        results[step] = game_acc
      else:
        print(f'\nWarning: Failed to parse accuracy table for step {step}')

    except FileNotFoundError:
      print('Executable ./build/chess/strength_chess not found.')
      sys.exit(1)

  if results:
    game_nums = sorted(list(next(iter(results.values())).keys()))
    with open('output.csv', 'w', newline='') as f:
      writer = csv.writer(f)
      writer.writerow(['step'] + game_nums)
      for step in sorted(results.keys()):
        writer.writerow([step] + [results[step][g] for g in game_nums])
    print('\nSuccessfully exported results to output.csv')

if __name__ == '__main__':
  model_directory = './chess_bt_b32_r8_p7_20bx256-7e7ac9/model'
  run_evaluations(model_directory, start_step=10000, max_step=100000, step_interval=10000)