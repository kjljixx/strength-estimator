#include "st_data_loader.h"
#include "environment.h"
#include "game_wrapper.h"
#include "random.h"
#include "rotation.h"
#include "sgf_loader.h"
#include "st_configuration.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace strength {

using namespace minizero;
using namespace minizero::utils;

namespace {

constexpr int kNumPhases = 3;
const char* const kPhaseNames[kNumPhases] = {"opening", "midgame", "endgame"};

struct PhaseRange {
  int phase_;
  int start_;
  int end_;
};

inline int playerBit(env::Player player) { return 1 << (static_cast<int>(player) - 1); }

std::vector<int> playerPositions(const EnvironmentLoader& env_loader, env::Player player, int start, int end)
{
  std::vector<int> positions;
  for (int j = start; j < end; ++j) {
    if (env_loader.getActionPairs()[j].first.getPlayer() == player) { positions.emplace_back(j); }
  }
  return positions;
}

// White (kPlayer1) moves at even indices and Black (kPlayer2) at odd ones.
int countPlayerPositions(env::Player player, int start, int end)
{
  const int parity = (player == env::Player::kPlayer1) ? 0 : 1;
  int count = 0;
  for (int j = start; j < end; ++j) { count += (j % 2 == parity); }
  return count;
}

std::string getRootTag(const std::string& root, const std::string& key)
{
  const size_t begin = root.find(key + "[");
  if (begin == std::string::npos) { return ""; }
  const size_t value_begin = begin + key.size() + 1;
  const size_t value_end = root.find(']', value_begin);
  return value_end == std::string::npos ? "" : root.substr(value_begin, value_end - value_begin);
}

int countMoves(const std::string& line)
{
  int count = 0;
  for (const char* marker : {";B[", ";W["}) {
    for (size_t pos = line.find(marker); pos != std::string::npos; pos = line.find(marker, pos + 1)) { ++count; }
  }
  return count;
}

// PM[] and PE[] hold the first move index of the midgame and endgame, or -1 when the game never reaches it.
std::string parsePhaseRanges(const std::string& root, int num_moves, std::vector<PhaseRange>& ranges)
{
  const std::string middle_tag = getRootTag(root, "PM");
  const std::string end_tag = getRootTag(root, "PE");
  if (middle_tag.empty() || end_tag.empty()) { return "missing_phase_tags"; }

  int middle = -1;
  int end = -1;
  try {
    middle = std::stoi(middle_tag);
    end = std::stoi(end_tag);
  } catch (const std::exception&) {
    return "invalid_phase_tags";
  }
  const bool middle_valid = middle >= -1 && middle < num_moves;
  const bool end_valid = end >= -1 && end < num_moves && (end == -1 || (middle != -1 && middle < end));
  if (!middle_valid || !end_valid) { return "invalid_phase_tags"; }

  std::vector<std::pair<int, int>> starts = {{0, 0}};
  if (middle >= 0) { starts.emplace_back(1, middle); }
  if (end >= 0) { starts.emplace_back(2, end); }
  ranges.clear();
  for (size_t i = 0; i < starts.size(); ++i) {
    int stop = (i + 1 < starts.size()) ? starts[i + 1].second : num_moves;
    if (stop > starts[i].second) { ranges.push_back({starts[i].first, starts[i].second, stop}); }
  }
  return "";
}

} // namespace

} // namespace

bool StDataLoaderThread::addEnvironmentLoader()
{
  std::string env_string = getSharedData()->getNextEnvString();
  if (env_string.empty()) { return false; }

  EnvironmentLoader env_loader = loadGame(env_string);
  if (!env_loader.getActionPairs().empty()) {
    assert(!env_loader.getTag("BR").empty());
    std::lock_guard<std::mutex> lock(getSharedData()->mutex_);
    if (strength::bt_use_win_chains) {
      // In win-chain lazy loading mode, games.txt is indexed not parsed here.
      // chain_games_ is no longer used; games are loaded on-demand per batch.
    } else {
      getSharedData()->env_loaders_map_[getRank(env_loader)].push_back(env_loader);
    }
  }
  return true;
}

bool StDataLoaderThread::sampleData()
{
  int batch_index = getSharedData()->getNextBatchIndex();
  if (batch_index >= config::learner_batch_size) { return false; }

  if (config::nn_type_name == "alphazero") {
    setAlphaZeroTrainingData(batch_index);
  } else if (config::nn_type_name == "rank") {
    setRankTrainingData(batch_index);
  } else if (config::nn_type_name == "bt") {
    setBTTrainingData(batch_index);
  } else {
    return false; // should not be here
  }

  return true;
}

const EnvironmentLoader& StDataLoaderThread::getEnvLoader(const GamePosition& gp)
{
  if (strength::bt_use_win_chains || strength::bt_use_phase_examples) {
    return getSharedData()->batch_chain_games_[gp.env_id_];
  }
  return getSharedData()->env_loaders_map_.at(gp.rank_)[gp.env_id_];
}

void StDataLoaderThread::setAlphaZeroTrainingData(int batch_index)
{
  // random pickup one position
  GamePosition gp = sampleTrainingData();

  // AlphaZero training data
  const EnvironmentLoader& env_loader = getEnvLoader(gp);
  Rotation rotation = static_cast<Rotation>(Random::randInt() % static_cast<int>(Rotation::kRotateSize));
  std::vector<float> features = calculateFeatures(env_loader, gp.pos_, rotation);
  std::vector<float> policy = env_loader.getPolicy(gp.pos_, rotation);
  std::vector<float> value = env_loader.getValue(gp.pos_);

  // write data to data_ptr
  std::copy(features.begin(), features.end(), getSharedData()->getDataPtr()->features_ + features.size() * batch_index);
  std::copy(policy.begin(), policy.end(), getSharedData()->getDataPtr()->policy_ + policy.size() * batch_index);
  std::copy(value.begin(), value.end(), getSharedData()->getDataPtr()->value_ + value.size() * batch_index);
}

void StDataLoaderThread::setRankTrainingData(int batch_index)
{
  // random pickup one position
  GamePosition gp = sampleTrainingData();

  // rankNet training data
  const EnvironmentLoader& env_loader = getEnvLoader(gp);
  Rotation rotation = static_cast<Rotation>(Random::randInt() % static_cast<int>(Rotation::kRotateSize));
  std::vector<float> features = calculateFeatures(env_loader, gp.pos_, rotation);
  std::vector<float> policy = env_loader.getPolicy(gp.pos_, rotation);
  std::vector<float> value = env_loader.getValue(gp.pos_);
  std::vector<float> rank(strength::nn_rank_size, 0.0f);
  rank[getSharedData()->rank_label_map_[gp.rank_]] = 1.0f;

  // write data to data_ptr
  std::copy(features.begin(), features.end(), getSharedData()->getDataPtr()->features_ + features.size() * batch_index);
  std::copy(policy.begin(), policy.end(), getSharedData()->getDataPtr()->policy_ + policy.size() * batch_index);
  std::copy(value.begin(), value.end(), getSharedData()->getDataPtr()->value_ + value.size() * batch_index);
  std::copy(rank.begin(), rank.end(), getSharedData()->getDataPtr()->rank_ + rank.size() * batch_index);
}

void StDataLoaderThread::setBTTrainingData(int batch_index)
{
  if (batch_index >= static_cast<int>(getSharedData()->bt_game_positions_.size())) {
    std::cerr << "WARN: setBTTrainingData batch_index " << batch_index
              << " >= bt_game_positions_ size " << getSharedData()->bt_game_positions_.size() << std::endl;
    return;
  }
  const GamePosition& gp = getSharedData()->bt_game_positions_[batch_index];

  // BT training data
  const EnvironmentLoader& env_loader = getEnvLoader(gp);
  Rotation rotation = static_cast<Rotation>(Random::randInt() % static_cast<int>(Rotation::kRotateSize));
  std::vector<float> features = calculateFeatures(env_loader, gp.pos_, rotation);
  std::vector<float> policy = env_loader.getPolicy(gp.pos_, rotation);
  std::vector<float> value = env_loader.getValue(gp.pos_);

  int num_rank_per_batch = strength::bt_num_rank_per_batch;
  int num_position_per_rank = strength::bt_num_position_per_rank;
  if ((batch_index % (num_rank_per_batch * num_position_per_rank)) < (num_position_per_rank)) {
    if (strength::bt_add_non_people) {
      Environment env;
      for (int i = 0; i < gp.pos_ - 1; ++i) {
        env.act(env_loader.getActionPairs()[i].first);
      }
      if (gp.pos_ > 0)
        env.setTurn(env_loader.getActionPairs()[gp.pos_ - 1].first.getPlayer());
      std::vector<Action> action_candidates;

      for (int action_id = 0; action_id < static_cast<int>(policy.size()); ++action_id) {
        Action action(action_id, env.getTurn());
        if (!env.isLegalAction(action)) { continue; }
        if (gp.pos_ != 0 && action_id == env_loader.getActionPairs()[gp.pos_ - 1].first.getActionID()) { continue; }
        action_candidates.push_back(action);
      }
      if (action_candidates.size() == 0) action_candidates.push_back(env_loader.getActionPairs()[gp.pos_ - 1].first);

      std::random_shuffle(action_candidates.begin(), action_candidates.end());
      env.act(action_candidates[0]);
      features = env.getFeatures(rotation);
      value[0] = -1.0f;
    }
  }
  // write data to data_ptr
  std::copy(features.begin(), features.end(), getSharedData()->getDataPtr()->features_ + features.size() * batch_index);
  std::copy(policy.begin(), policy.end(), getSharedData()->getDataPtr()->policy_ + policy.size() * batch_index);
  std::copy(value.begin(), value.end(), getSharedData()->getDataPtr()->value_ + value.size() * batch_index);
  getSharedData()->getDataPtr()->rank_[batch_index] = gp.rank_;
}

GamePosition StDataLoaderThread::sampleTrainingData()
{
  GamePosition gp;
  if (strength::bt_use_win_chains) {
    if (getSharedData()->batch_chain_games_.empty()) {
      std::cerr << "WARN: sampleTrainingData called with empty batch_chain_games_ (should not happen for BT win-chains)" << std::endl;
      return GamePosition(0, 0, 0);
    }
    gp.rank_ = 0;
    gp.env_id_ = Random::randInt() % getSharedData()->batch_chain_games_.size();
    gp.pos_ = Random::randInt() % getSharedData()->batch_chain_games_[gp.env_id_].getActionPairs().size();
    return gp;
  }
  auto it = getSharedData()->env_loaders_map_.begin();
  std::advance(it, Random::randInt() % getSharedData()->env_loaders_map_.size());
  gp.rank_ = it->first;
  gp.env_id_ = Random::randInt() % getSharedData()->env_loaders_map_[gp.rank_].size();
  gp.pos_ = Random::randInt() % getSharedData()->env_loaders_map_[gp.rank_][gp.env_id_].getActionPairs().size();
  return gp;
}

StDataLoader::StDataLoader(const std::string& conf_file_name)
  : learner::DataLoader("")
{
  minizero::env::setUpEnv();
  minizero::config::ConfigureLoader cl;
  strength::setConfiguration(cl);
  cl.loadFromFile(conf_file_name);
}

void StDataLoader::initialize()
{
  DataLoader::initialize();
  int seed = config::program_auto_seed ? std::random_device()() : config::program_seed;
  Random::seed(seed);
}

void StDataLoader::indexChainGamesFile(const std::string& file_name)
{
  std::ifstream fin(file_name, std::ios::binary);
  if (!fin) {
    std::cerr << "ERROR: cannot open chain games file " << file_name << std::endl;
    return;
  }

  getSharedData()->chain_games_file_ = file_name;
  getSharedData()->chain_game_offsets_.clear();

  std::streamoff offset = 0;
  std::string line;
  while (std::getline(fin, line)) {
    getSharedData()->chain_game_offsets_.push_back(offset);
    offset = fin.tellg();
  }

  std::cerr << "=== lazy win-chain loader ===" << std::endl;
  std::cerr << "games_file=" << file_name << std::endl;
  std::cerr << "indexed_games=" << getSharedData()->chain_game_offsets_.size() << std::endl;
  std::cerr << "offset_index_bytes=" << (getSharedData()->chain_game_offsets_.size() * sizeof(std::streamoff)) << std::endl;

  // Validate index
  if (getSharedData()->chain_game_offsets_.empty()) {
    std::cerr << "ERROR: no games indexed from " << file_name << std::endl;
    return;
  }

  // Spot-check: verify games at start, middle, end can be read and parsed
  std::vector<size_t> check_indices = {0};
  if (getSharedData()->chain_game_offsets_.size() > 1) {
    check_indices.push_back(getSharedData()->chain_game_offsets_.size() / 2);
    check_indices.push_back(getSharedData()->chain_game_offsets_.size() - 1);
  }
  for (size_t idx : check_indices) {
    std::ifstream fcheck(file_name, std::ios::binary);
    fcheck.seekg(getSharedData()->chain_game_offsets_[idx]);
    std::string check_line;
    std::getline(fcheck, check_line);
    if (check_line.empty()) {
      std::cerr << "WARN: empty line at index " << idx << std::endl;
      continue;
    }
    EnvironmentLoader test_loader = loadGame(check_line);
    if (test_loader.getActionPairs().empty()) {
      std::cerr << "WARN: parsed game at index " << idx << " has no action pairs" << std::endl;
    }
  }
}

EnvironmentLoader StDataLoader::loadChainGame(int game_id)
{
  const auto& offsets = getSharedData()->chain_game_offsets_;
  if (game_id < 0 || game_id >= static_cast<int>(offsets.size())) {
    std::cerr << "ERROR: game_id " << game_id << " out of range [0, " << offsets.size() << ")" << std::endl;
    return EnvironmentLoader();
  }

  std::ifstream fin(getSharedData()->chain_games_file_, std::ios::binary);
  if (!fin) {
    std::cerr << "ERROR: cannot open chain games file " << getSharedData()->chain_games_file_ << std::endl;
    return EnvironmentLoader();
  }

  fin.seekg(offsets[game_id]);
  std::string line;
  std::getline(fin, line);

  if (line.empty()) {
    std::cerr << "ERROR: empty line at game_id " << game_id << std::endl;
    return EnvironmentLoader();
  }

  EnvironmentLoader env_loader = loadGame(line);
  if (env_loader.getActionPairs().empty()) {
    std::cerr << "WARN: parsed game " << game_id << " has no action pairs" << std::endl;
  }
  return env_loader;
}

void StDataLoader::loadDataFromFile(const std::string& file_name)
{
  if (strength::bt_use_phase_examples && strength::bt_use_win_chains) {
    throw std::runtime_error("bt_use_phase_examples cannot be combined with bt_use_win_chains");
  }
  if (strength::bt_use_phase_examples && strength::bt_add_non_people) {
    throw std::runtime_error("bt_use_phase_examples cannot be combined with bt_add_non_people");
  }
  if (strength::bt_use_phase_examples) {
    indexPhaseGamesFile(file_name);
    return;
  }
  if (strength::bt_use_win_chains) {
    indexChainGamesFile(file_name);

    int label = 0;
    getSharedData()->rank_label_map_.clear();
    for (int i = 0; i < strength::bt_num_rank_per_batch; ++i) {
      getSharedData()->rank_label_map_[i] = label++;
    }
    return;
  }

  DataLoader::loadDataFromFile(file_name);

  int label = 0;
  getSharedData()->rank_label_map_.clear();
  for (auto& m : getSharedData()->env_loaders_map_) { getSharedData()->rank_label_map_[m.first] = label++; }
}

void StDataLoader::indexPhaseGamesFile(const std::string& file_name)
{
  std::ifstream fin(file_name, std::ios::binary);
  if (!fin) { throw std::runtime_error("cannot open phase games file " + file_name); }

  const int num_positions = strength::bt_num_position_per_rank;
  const int file_index = static_cast<int>(getSharedData()->phase_files_.size());
  getSharedData()->phase_files_.push_back(file_name);

  std::vector<PhaseRange> ranges;
  int indexed_games = 0;
  int rank_out_of_range = 0;
  std::streamoff offset = 0;
  for (std::string line; std::getline(fin, line); offset = fin.tellg()) {
    if (line.empty()) { continue; }
    const size_t first_move = std::min(line.find(";B[", 1), line.find(";W[", 1));
    const std::string root = line.substr(0, first_move);
    int rank = -1;
    try {
      rank = (std::stoi(getRootTag(root, "BR")) - strength::nn_rank_min_elo) / strength::nn_rank_elo_interval;
    } catch (const std::exception&) {
      rank = -1;
    }
    if (first_move == std::string::npos || rank < 0 || rank >= strength::nn_rank_size || line.compare(first_move, 3, ";B[") != 0) {
      ++rank_out_of_range;
      continue;
    }

    PhaseBinStats& stats = getSharedData()->phase_stats_[rank];
    ++stats.games_;
    const std::string error = parsePhaseRanges(root, countMoves(line), ranges);
    if (!error.empty()) {
      (error == "missing_phase_tags" ? stats.missing_tags_ : stats.invalid_tags_)++;
      continue;
    }

    const int game_id = static_cast<int>(getSharedData()->phase_game_refs_.size());
    bool game_has_example = false;
    for (const PhaseRange& range : ranges) {
      PhaseExample example;
      example.game_id_ = game_id;
      example.phase_ = range.phase_;
      example.start_ = range.start_;
      example.end_ = range.end_;
      for (env::Player player : {env::Player::kPlayer1, env::Player::kPlayer2}) {
        if (countPlayerPositions(player, range.start_, range.end_) >= num_positions) { example.eligible_players_ |= playerBit(player); }
      }
      if (example.eligible_players_ == 0) {
        ++stats.too_few_positions_[range.phase_];
        continue;
      }
      getSharedData()->phase_examples_map_[rank].push_back(example);
      ++stats.examples_[range.phase_];
      game_has_example = true;
    }
    if (game_has_example) {
      PhaseGameRef ref;
      ref.file_index_ = file_index;
      ref.offset_ = offset;
      getSharedData()->phase_game_refs_.push_back(ref);
      ++indexed_games;
    }
  }
  std::cerr << "indexed phase games file=" << file_name << " games_with_examples=" << indexed_games
            << " skipped_unusable_rank_or_first_player=" << rank_out_of_range << std::endl;
}

EnvironmentLoader StDataLoader::loadPhaseGame(int game_id)
{
  const PhaseGameRef& ref = getSharedData()->phase_game_refs_.at(game_id);
  std::ifstream fin(getSharedData()->phase_files_.at(ref.file_index_), std::ios::binary);
  fin.seekg(ref.offset_);
  std::string line;
  std::getline(fin, line);
  EnvironmentLoader env_loader = loadGame(line);
  if (env_loader.getActionPairs().empty()) { throw std::runtime_error("phase game " + std::to_string(game_id) + " failed to load"); }
  return env_loader;
}

void StDataLoader::finalizePhaseExamples()
{
  const int num_bins = strength::nn_rank_size;
  const int num_positions = strength::bt_num_position_per_rank;
  const int batch_positions = strength::bt_num_batch_size * strength::bt_num_rank_per_batch * num_positions;
  std::cerr << "=== phase example loader ===" << std::endl;
  std::cerr << "elo_bins=" << num_bins << " min_elo=" << strength::nn_rank_min_elo
            << " elo_interval=" << strength::nn_rank_elo_interval
            << " bt_num_batch_size=" << strength::bt_num_batch_size
            << " bt_num_rank_per_batch=" << strength::bt_num_rank_per_batch
            << " positions_per_example=" << num_positions
            << " positions_per_step=" << batch_positions
            << " learner_batch_size=" << config::learner_batch_size
            << " indexed_games=" << getSharedData()->phase_game_refs_.size() << std::endl;

  if (batch_positions != config::learner_batch_size) {
    throw std::runtime_error("phase examples need learner_batch_size == bt_num_batch_size * bt_num_rank_per_batch * bt_num_position_per_rank");
  }
  if (strength::bt_num_rank_per_batch > num_bins) {
    throw std::runtime_error("bt_num_rank_per_batch cannot exceed nn_rank_size");
  }

  std::vector<int> empty_bins;
  for (int bin = 0; bin < num_bins; ++bin) {
    const int low_elo = strength::nn_rank_min_elo + bin * strength::nn_rank_elo_interval;
    const PhaseBinStats& stats = getSharedData()->phase_stats_[bin];
    std::cerr << "bin=" << bin << " elo=" << low_elo << "-" << (low_elo + strength::nn_rank_elo_interval - 1)
              << " games=" << stats.games_;
    for (int phase = 0; phase < kNumPhases; ++phase) {
      std::cerr << " " << kPhaseNames[phase] << "=" << stats.examples_[phase]
                << "(skipped_fewer_than_" << num_positions << "_positions=" << stats.too_few_positions_[phase] << ")";
    }
    std::cerr << " skipped_missing_phase_tags=" << stats.missing_tags_
              << " skipped_invalid_phase_tags=" << stats.invalid_tags_ << std::endl;
    if (getSharedData()->phase_examples_map_[bin].empty()) { empty_bins.push_back(bin); }
  }
  if (!empty_bins.empty()) {
    std::string message = "no eligible phase examples for elo bins:";
    for (int bin : empty_bins) { message += " " + std::to_string(bin); }
    throw std::runtime_error(message);
  }
}

void StDataLoader::loadWinChainsFromFile(const std::string& file_name)
{
  getSharedData()->win_chains_.clear();
  std::ifstream fin(file_name);
  if (!fin) {
    std::cerr << "ERROR: cannot open win chains file " << file_name << std::endl;
    return;
  }

  int line_no = 0;
  int skipped = 0;
  for (std::string line; std::getline(fin, line);) {
    ++line_no;
    if (line.empty()) { continue; }
    std::istringstream iss(line);
    std::vector<WinChainSlot> chain;
    std::string token;
    bool ok = true;
    while (iss >> token) {
      size_t colon = token.find(':');
      if (colon == std::string::npos) {
        std::cerr << "WARN: bad chain token '" << token << "' at line " << line_no << std::endl;
        ok = false;
        break;
      }
      int game_id = std::stoi(token.substr(0, colon));
      char color = token[colon + 1];
      if (game_id < 0 || game_id >= static_cast<int>(getSharedData()->chain_game_offsets_.size())) {
        std::cerr << "WARN: game_id " << game_id << " out of range at line " << line_no << std::endl;
        ok = false;
        break;
      }
      WinChainSlot slot;
      slot.game_id_ = game_id;
      if (color == 'W' || color == 'w') {
        slot.player_ = static_cast<int>(env::Player::kPlayer1);
      } else if (color == 'B' || color == 'b') {
        slot.player_ = static_cast<int>(env::Player::kPlayer2);
      } else {
        std::cerr << "WARN: bad color '" << color << "' at line " << line_no << std::endl;
        ok = false;
        break;
      }
      chain.push_back(slot);
    }
    if (!ok) {
      ++skipped;
      continue;
    }
    if (static_cast<int>(chain.size()) != strength::bt_num_rank_per_batch) {
      std::cerr << "WARN: chain length " << chain.size() << " != bt_num_rank_per_batch "
                << strength::bt_num_rank_per_batch << " at line " << line_no << std::endl;
      ++skipped;
      continue;
    }
    getSharedData()->win_chains_.push_back(std::move(chain));
  }
  std::cerr << "loaded win chains=" << getSharedData()->win_chains_.size()
            << " skipped=" << skipped << " from " << file_name << std::endl;
}

void StDataLoader::sampleData()
{
  if (config::nn_type_name == "bt") {
    if (strength::bt_use_phase_examples && !phase_examples_built_) {
      finalizePhaseExamples();
      phase_examples_built_ = true;
    }
    allocateBTGamePositions();
  }
  DataLoader::sampleData();
}

void StDataLoader::selectBTWinChains(std::vector<const std::vector<WinChainSlot>*>& selected_chains)
{
  const auto& chains = getSharedData()->win_chains_;
  selected_chains.clear();
  selected_chains.reserve(strength::bt_num_batch_size);

  for (int batch_index = 0; batch_index < strength::bt_num_batch_size; ++batch_index) {
    const auto& chain = chains[Random::randInt() % chains.size()];
    selected_chains.push_back(&chain);
  }
}

void StDataLoader::loadBTBatchGames(const std::vector<const std::vector<WinChainSlot>*>& selected_chains)
{
  // Clear previous batch
  getSharedData()->batch_chain_games_.clear();
  getSharedData()->batch_game_id_map_.clear();

  // Collect unique game IDs from selected chains
  std::unordered_set<int> unique_game_ids;
  for (const auto* chain : selected_chains) {
    for (const auto& slot : *chain) {
      unique_game_ids.insert(slot.game_id_);
    }
  }

  // Load each unique game
  getSharedData()->batch_chain_games_.reserve(unique_game_ids.size());
  int local_idx = 0;
  for (int game_id : unique_game_ids) {
    EnvironmentLoader env_loader = loadChainGame(game_id);
    if (!env_loader.getActionPairs().empty()) {
      getSharedData()->batch_chain_games_.push_back(std::move(env_loader));
      getSharedData()->batch_game_id_map_[game_id] = local_idx++;
    } else {
      std::cerr << "WARN: game " << game_id << " failed to load, skipping" << std::endl;
    }
  }

  std::cerr << "BT lazy batch: selected_chains=" << selected_chains.size()
            << " game_references=" << (selected_chains.size() * strength::bt_num_rank_per_batch)
            << " unique_games=" << unique_game_ids.size()
            << " parsed_games=" << getSharedData()->batch_chain_games_.size() << std::endl;
}

void StDataLoader::allocateBTWinChainPositionsFromLoaded(const std::vector<const std::vector<WinChainSlot>*>& selected_chains)
{
  getSharedData()->bt_game_positions_.clear();
  const int num_rank = strength::bt_num_rank_per_batch;
  const int num_pos = strength::bt_num_position_per_rank;
  const int expected = strength::bt_num_batch_size * num_rank * num_pos;

  for (const auto* chain : selected_chains) {
    for (int slot = 0; slot < num_rank; ++slot) {
      const WinChainSlot& cs = (*chain)[slot];
      auto it = getSharedData()->batch_game_id_map_.find(cs.game_id_);
      if (it == getSharedData()->batch_game_id_map_.end()) {
        std::cerr << "ERROR: game " << cs.game_id_ << " not loaded for batch; this should not happen" << std::endl;
        // Don't create dummy positions - this would corrupt training data
        // The game should have been loaded in loadBTBatchGames
        continue;
      }
      int local_game_id = it->second;
      const EnvironmentLoader& env_loader = getSharedData()->batch_chain_games_[local_game_id];
      env::Player player = static_cast<env::Player>(cs.player_);
      std::vector<int> positions;
      for (size_t j = 0; j < env_loader.getActionPairs().size(); ++j) {
        if (env_loader.getActionPairs()[j].first.getPlayer() == player) { positions.emplace_back(static_cast<int>(j)); }
      }
      if (positions.empty()) {
        std::cerr << "WARN: no moves for player " << cs.player_ << " in game " << cs.game_id_
                  << "; using all moves" << std::endl;
        for (size_t j = 0; j < env_loader.getActionPairs().size(); ++j) {
          positions.emplace_back(static_cast<int>(j));
        }
      }
      if (positions.empty()) {
        std::cerr << "WARN: empty game " << cs.game_id_ << "; skipping slot fill with pos 0" << std::endl;
        for (int j = 0; j < num_pos; ++j) {
          getSharedData()->bt_game_positions_.emplace_back(GamePosition(slot, local_game_id, 0));
        }
        continue;
      }
      std::random_shuffle(positions.begin(), positions.end());
      for (int j = 0; j < num_pos; ++j) {
        int pos = positions[j % positions.size()];
        // rank_ = slot index (weak=0 ... strong=K-1); env_id_ = batch-local index in batch_chain_games_
        getSharedData()->bt_game_positions_.emplace_back(GamePosition(slot, local_game_id, pos));
      }
    }
  }
  if (static_cast<int>(getSharedData()->bt_game_positions_.size()) != expected) {
    std::cerr << "WARN: bt_game_positions size "
              << getSharedData()->bt_game_positions_.size() << " != expected " << expected << std::endl;
  }
}

void StDataLoader::allocateBTWinChainPositions()
{
  const auto& chains = getSharedData()->win_chains_;
  if (chains.empty() || getSharedData()->chain_game_offsets_.empty()) {
    std::cerr << "ERROR: allocateBTWinChainPositions with no chains/games loaded"
              << " chains=" << chains.size()
              << " indexed_games=" << getSharedData()->chain_game_offsets_.size() << std::endl;
    return;
  }

  std::vector<const std::vector<WinChainSlot>*> selected_chains;
  selectBTWinChains(selected_chains);
  loadBTBatchGames(selected_chains);
  allocateBTWinChainPositionsFromLoaded(selected_chains);
}

void StDataLoader::allocateBTGamePositions()
{
  if (strength::bt_use_win_chains) {
    allocateBTWinChainPositions();
    return;
  }
  if (strength::bt_use_phase_examples) {
    allocateBTPhaseExamplePositions();
    return;
  }

  getSharedData()->bt_game_positions_.clear();

  for (int batch_index = 0; batch_index < strength::bt_num_batch_size; batch_index++) {
    // sample ranks
    std::vector<int> ranks;
    for (auto& m : getSharedData()->env_loaders_map_) { ranks.emplace_back(m.first); }
    std::random_shuffle(ranks.begin(), ranks.end());
    int num_rank = (strength::bt_num_rank_per_batch > 0
                      ? std::min(strength::bt_num_rank_per_batch, static_cast<int>(getSharedData()->rank_label_map_.size()))
                      : Random::randInt() % getSharedData()->env_loaders_map_.size());
    ranks.resize(num_rank);

    if (strength::bt_add_non_people) {
      num_rank = (strength::bt_num_rank_per_batch > 0
                    ? std::min(strength::bt_num_rank_per_batch, static_cast<int>(getSharedData()->rank_label_map_.size()) + 1)
                    : Random::randInt() % getSharedData()->env_loaders_map_.size());
    }
    if (strength::bt_add_non_people) {
      if (num_rank > static_cast<int>(getSharedData()->rank_label_map_.size())) {
        int randomIndex = Random::randInt() % getSharedData()->env_loaders_map_.size();
        int count = 0;
        for (auto& m : getSharedData()->env_loaders_map_) {
          if (count == randomIndex) {
            ranks.insert(ranks.begin(), m.first);
            break;
          }
          count++;
        }
      }
      std::sort(ranks.begin() + 1, ranks.end());
    } else {
      std::sort(ranks.begin(), ranks.end());
    }
    // sample positions
    for (int i = 0; i < num_rank; ++i) {
      int rank = ranks[i];

      if (strength::bt_use_same_game_per_rank) {
        int env_id = Random::randInt() % getSharedData()->env_loaders_map_[rank].size();
        const EnvironmentLoader& env_loader = getSharedData()->env_loaders_map_[rank][env_id];
        env::Player player = (Random::randInt() % 2 == 0 ? env::Player::kPlayer1 : env::Player::kPlayer2);
        std::vector<int> positions;
        for (size_t j = 0; j < env_loader.getActionPairs().size(); ++j) {
          if (env_loader.getActionPairs()[j].first.getPlayer() == player) { positions.emplace_back(j); }
        }
        std::random_shuffle(positions.begin(), positions.end());
        for (int j = 0; j < strength::bt_num_position_per_rank; ++j) {
          getSharedData()->bt_game_positions_.emplace_back(GamePosition(rank, env_id, positions[j]));
        }
      } else {
        for (int j = 0; j < strength::bt_num_position_per_rank; ++j) {
          int env_id = Random::randInt() % getSharedData()->env_loaders_map_[rank].size();
          int pos = Random::randInt() % getSharedData()->env_loaders_map_[rank][env_id].getActionPairs().size();
          getSharedData()->bt_game_positions_.emplace_back(GamePosition(rank, env_id, pos));
        }
      }
    }
  }
}

void StDataLoader::allocateBTPhaseExamplePositions()
{
  const int num_bins = strength::nn_rank_size;
  const int num_rank = strength::bt_num_rank_per_batch;
  const int num_pos = strength::bt_num_position_per_rank;

  std::vector<std::vector<int>> group_ranks(strength::bt_num_batch_size);
  std::vector<std::vector<const PhaseExample*>> group_examples(strength::bt_num_batch_size);
  std::unordered_set<int> unique_game_ids;
  for (int group = 0; group < strength::bt_num_batch_size; ++group) {
    std::vector<int>& ranks = group_ranks[group];
    for (int bin = 0; bin < num_bins; ++bin) { ranks.push_back(bin); }
    std::random_shuffle(ranks.begin(), ranks.end());
    ranks.resize(num_rank);
    std::sort(ranks.begin(), ranks.end());
    for (int rank : ranks) {
      const std::vector<PhaseExample>& pool = getSharedData()->phase_examples_map_.at(rank);
      const PhaseExample* example = &pool[Random::randInt() % pool.size()];
      group_examples[group].push_back(example);
      unique_game_ids.insert(example->game_id_);
    }
  }

  getSharedData()->batch_chain_games_.clear();
  getSharedData()->batch_game_id_map_.clear();
  getSharedData()->batch_chain_games_.reserve(unique_game_ids.size());
  for (int game_id : unique_game_ids) {
    getSharedData()->batch_game_id_map_[game_id] = static_cast<int>(getSharedData()->batch_chain_games_.size());
    getSharedData()->batch_chain_games_.push_back(loadPhaseGame(game_id));
  }

  getSharedData()->bt_game_positions_.clear();
  for (int group = 0; group < strength::bt_num_batch_size; ++group) {
    for (int slot = 0; slot < num_rank; ++slot) {
      const PhaseExample& example = *group_examples[group][slot];
      const int local_game_id = getSharedData()->batch_game_id_map_.at(example.game_id_);
      const EnvironmentLoader& env_loader = getSharedData()->batch_chain_games_[local_game_id];
      if (example.end_ > static_cast<int>(env_loader.getActionPairs().size())) {
        throw std::runtime_error("phase example of game " + std::to_string(example.game_id_) + " extends past the end of the game");
      }

      std::vector<env::Player> eligible_players;
      for (env::Player player : {env::Player::kPlayer1, env::Player::kPlayer2}) {
        if (example.eligible_players_ & playerBit(player)) { eligible_players.push_back(player); }
      }
      env::Player player = eligible_players[Random::randInt() % eligible_players.size()];

      std::vector<int> positions = playerPositions(env_loader, player, example.start_, example.end_);
      if (static_cast<int>(positions.size()) < num_pos) {
        throw std::runtime_error("phase example of game " + std::to_string(example.game_id_) + " has too few positions for the selected player");
      }
      std::random_shuffle(positions.begin(), positions.end());
      for (int j = 0; j < num_pos; ++j) {
        getSharedData()->bt_game_positions_.emplace_back(GamePosition(group_ranks[group][slot], local_game_id, positions[j]));
      }
    }
  }

  const size_t expected = static_cast<size_t>(strength::bt_num_batch_size) * num_rank * num_pos;
  if (getSharedData()->bt_game_positions_.size() != expected) {
    throw std::runtime_error("phase example sampler produced " + std::to_string(getSharedData()->bt_game_positions_.size()) +
                             " positions, expected " + std::to_string(expected));
  }
}

} // namespace strength
