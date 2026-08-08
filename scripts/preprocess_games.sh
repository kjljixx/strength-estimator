#!/bin/bash

# Usage:
#   ./scripts/preprocess_games.sh           # Elo-bin pipeline (default)
#   ./scripts/preprocess_games.sh elo
#   ./scripts/preprocess_games.sh chain     # win-chain ordinal data (requires training_sgf/)
#   ./scripts/preprocess_games.sh both

MODE="${1:-elo}"

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


run_elo_pipeline() {
  cp ./scripts/sgf_filter_random_sample.py ./
  python3 sgf_filter_random_sample.py

  cp ./scripts/random_sample.py ./
  python3 random_sample.py

  declare -A folder_map
  folder_map["rank_50000_1000_2600_200interval/train"]="training_sgf_chess"
  folder_map["rank_50000_1000_2600_200interval/test"]="query_sgf_chess"
  folder_map["rank_50000_1000_2600_200interval/cand"]="candidate_sgf_chess"

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
  echo "Elo-bin files merged into training/query/candidate dirs."
}

run_chain_pipeline() {
  echo "Building win-chain ordinal data into training_sgf_chess_chain/"
  python3 ./scripts/sgf_filter_win_chain.py \
    --input-dir training_sgf \
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
  chain)
    run_chain_pipeline
    ;;
  both)
    run_elo_pipeline
    run_chain_pipeline
    ;;
  *)
    echo "Unknown mode: $MODE (use elo|chain|both)"
    exit 1
    ;;
esac

echo "Preprocess complete (mode=$MODE)."
