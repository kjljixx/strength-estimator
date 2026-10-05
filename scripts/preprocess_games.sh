#!/bin/bash

# Usage:
#   ./scripts/preprocess_games.sh           # Elo-bin pipeline (default)
#   ./scripts/preprocess_games.sh elo
#   ./scripts/preprocess_games.sh chain     # Elo split first, then win-chains from train only
#   ./scripts/preprocess_games.sh both      # same as chain (Elo split + train-only win-chains)
#
# Options (after the mode):
#   --from-sgf FILE         start from an existing SGF file or directory instead of downloading/converting PGN
#   --elo-interval N        Elo bin width (default 200)
#   --phase-examples        tag training games with opening/midgame/endgame boundaries
#   --output-root DIR       write everything under DIR instead of the repository root
#   --held-out-dir DIR      held-out SGFs to exclude from training (repeatable; default candidate_sgf_chess query_sgf_chess)

MODE="${1:-elo}"
shift 2>/dev/null

FROM_SGF=""
ELO_INTERVAL=200
PHASE_FLAG=""
OUTPUT_ROOT=""
HELD_OUT_DIRS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --from-sgf) FROM_SGF="$2"; shift 2 ;;
    --elo-interval) ELO_INTERVAL="$2"; shift 2 ;;
    --phase-examples) PHASE_FLAG="--phase-examples"; shift ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --held-out-dir) HELD_OUT_DIRS+=("$2"); shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

if [[ -n "$OUTPUT_ROOT" && "$MODE" != "elo" ]]; then
  echo "ERROR: --output-root only supports the elo mode" >&2
  exit 1
fi
echo "Preprocess config: mode=$MODE from_sgf=${FROM_SGF:-<download>} elo_interval=$ELO_INTERVAL phase_examples=${PHASE_FLAG:-off} output_root=${OUTPUT_ROOT:-<repo root>}"

if [[ -z "$FROM_SGF" ]]; then
  cp ./scripts/data.py download_chess_game/
  cp ./scripts/board.py download_chess_game/
  cd download_chess_game/

  year=2024
  month=01
  mkdir -p "database${year}/${year}${month}/"
  python3 data.py $year $month -u > "database${year}/${year}${month}/${year}-${month}-convert.txt"
  cd ../

  mkdir -p training_sgf
  mv "download_chess_game/database${year}/${year}${month}/${year}-${month}-convert.txt" training_sgf/
fi


run_elo_pipeline() {
  local split_root="rank_50000_1000_2600_${ELO_INTERVAL}interval"
  local train_target="training_chess_chain_raw"
  local query_target="query_sgf_chess"
  local cand_target="candidate_sgf_chess"
  local filter_args=(--input "${FROM_SGF:-training_sgf}" --elo-interval "$ELO_INTERVAL" $PHASE_FLAG)

  if [[ -n "$OUTPUT_ROOT" ]]; then
    split_root="$OUTPUT_ROOT/split"
    train_target="$OUTPUT_ROOT/training_chess"
    query_target="$OUTPUT_ROOT/query_sgf_chess"
    cand_target="$OUTPUT_ROOT/candidate_sgf_chess"
    if [[ ${#HELD_OUT_DIRS[@]} -eq 0 ]]; then HELD_OUT_DIRS=(candidate_sgf_chess query_sgf_chess); fi
    for dir in "${HELD_OUT_DIRS[@]}"; do
      if [[ ! -d "$dir" ]]; then echo "ERROR: held-out directory $dir missing" >&2; exit 1; fi
      filter_args+=(--held-out-dir "$dir")
    done
  fi
  filter_args+=(--output-root "$split_root")

  python3 ./scripts/sgf_filter_random_sample.py "${filter_args[@]}" || exit 1
  python3 ./scripts/random_sample.py --root "$split_root" || exit 1

  declare -A folder_map
  folder_map["$split_root/train"]="$train_target"
  folder_map["$split_root/test"]="$query_target"
  folder_map["$split_root/cand"]="$cand_target"

  for source_dir in "${!folder_map[@]}"; do
    target_dir="${folder_map[$source_dir]}"
    mkdir -p "$target_dir"
    echo "Processing $source_dir -> $target_dir"
    for folder in "$source_dir"/sgf_*; do
      if [[ -d "$folder" ]]; then
        base_name=$(basename "$folder")
        new_name="${base_name#sgf_}.txt"
        cat "$folder"/*.txt > "$target_dir/$new_name"
        echo "Merged $folder/*.txt -> $target_dir/$new_name"
      fi
    done
  done
  echo "Elo-bin files merged into $train_target, $query_target, $cand_target."
}

run_chain_pipeline() {
  if [[ ! -d training_chess_chain_raw ]] || [[ -z "$(ls -A training_chess_chain_raw 2>/dev/null)" ]]; then
    echo "ERROR: training_chess_chain_raw/ missing or empty; run Elo split first" >&2
    exit 1
  fi
  echo "Building win-chain ordinal data from training_chess_chain_raw/ into training_sgf_chess_chain/"
  python3 ./scripts/sgf_filter_win_chain.py \
    --input-dir training_chess_chain_raw \
    --output-dir training_sgf_chess_chain \
    --chain-length 8 \
    --max-chains 50000 \
    --max-games-per-player 32 \
    --seed 0
  echo "Win-chain data written to training_sgf_chess_chain/{games.txt,chains.txt}"
}

case "$MODE" in
  elo)
    run_elo_pipeline
    ;;
  chain|both)
    run_elo_pipeline
    run_chain_pipeline
    ;;
  *)
    echo "Unknown mode: $MODE (use elo|chain|both)"
    exit 1
    ;;
esac

echo "Preprocess complete (mode=$MODE)."
