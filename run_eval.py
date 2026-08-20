import csv
import re
import subprocess
import sys

def parse_evaluator_output(output):
  game_acc = {}
  recording = False

  for line in output.splitlines():
    if 'Summarizing accuracy' in line:
      recording = True
      continue

    if recording:
      tokens = line.strip().split()
      if len(tokens) >= 10 and tokens[0].isdigit():
        game_num = int(tokens[0])
        if 1 <= game_num <= 100:
          game_acc[game_num] = float(tokens[9])

  return game_acc

def run_evaluations(model_dir, start_step=10000, max_step=100000, step_interval=10000):
  results = {}

  for step in range(start_step, max_step + 1, step_interval):
    weight_file = f'{model_dir}/weight_iter_{step}.pt'

    cmd = [
      './build/chess/strength_chess',
      '-conf_file', 'cfg/se_chess.cfg',
      '-mode', 'evaluator',
      '-conf_str', f'nn_file_name={weight_file}'
    ]

    print(f'Running evaluation for step {step}...')
    try:
      proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
      parsed = parse_evaluator_output(proc.stdout)
      if parsed:
        results[step] = parsed
      else:
        print(f'Warning: Failed to parse accuracy table for step {step}')
    except subprocess.CalledProcessError as e:
      print(f'Error executing step {step}: {e}')
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
    print('Successfully exported results to output.csv')

if __name__ == '__main__':
  model_directory = './chess_bt_b32_r8_p7_20bx256-7e7ac9/model'
  run_evaluations(model_directory, start_step=10000, max_step=100000, step_interval=10000)