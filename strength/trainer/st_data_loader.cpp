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

struct PhaseCounts {
  int games_ = 0;
  int missing_tags_ = 0;
  int invalid_tags_ = 0;
  int examples_[kNumPhases] = {};
  int too_few_positions_[kNumPhases] = {};
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

// PM[] and PE[] hold the first move index of the midgame and endgame, or -1 when the game never reaches it.
std::string parsePhaseRanges(const EnvironmentLoader& env_loader, std::vector<PhaseRange>& ranges)
{
  const std::string middle_tag = env_loader.getTag("PM");
  const std::string end_tag = env_loader.getTag("PE");
  if (middle_tag.empty() || end_tag.empty()) { return "missing_phase_tags"; }

  int middle = -1;
  int end = -1;
  try {
    middle = std::stoi(middle_tag);
    end = std::stoi(end_tag);
  } catch (const std::exception&) {
    return "invalid_phase_tags";
  }
  const int num_moves = static_cast<int>(env_loader.getActionPairs().size());
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
  if (strength::bt_use_win_chains) {
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
  if (strength::bt_use_phase_examples) { buildPhaseExamples(); }
}

void StDataLoader::buildPhaseExamples()
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
            << " learner_batch_size=" << config::learner_batch_size << std::endl;

  if (batch_positions != config::learner_batch_size) {
    throw std::runtime_error("phase examples need learner_batch_size == bt_num_batch_size * bt_num_rank_per_batch * bt_num_position_per_rank");
  }
  if (strength::bt_num_rank_per_batch > num_bins) {
    throw std::runtime_error("bt_num_rank_per_batch cannot exceed nn_rank_size");
  }

  getSharedData()->phase_examples_map_.clear();
  std::map<int, PhaseCounts> counts;
  std::vector<PhaseRange> ranges;
  for (const auto& bin : getSharedData()->env_loaders_map_) {
    if (bin.first < 0 || bin.first >= num_bins) {
      throw std::runtime_error("games found in elo bin " + std::to_string(bin.first) + " outside [0, nn_rank_size)");
    }
    PhaseCounts& bin_counts = counts[bin.first];
    std::vector<PhaseExample>& examples = getSharedData()->phase_examples_map_[bin.first];
    for (size_t env_id = 0; env_id < bin.second.size(); ++env_id) {
      const EnvironmentLoader& env_loader = bin.second[env_id];
      ++bin_counts.games_;
      const std::string error = parsePhaseRanges(env_loader, ranges);
      if (!error.empty()) {
        (error == "missing_phase_tags" ? bin_counts.missing_tags_ : bin_counts.invalid_tags_)++;
        continue;
      }
      for (const PhaseRange& range : ranges) {
        PhaseExample example;
        example.env_id_ = static_cast<int>(env_id);
        example.phase_ = range.phase_;
        example.start_ = range.start_;
        example.end_ = range.end_;
        for (env::Player player : {env::Player::kPlayer1, env::Player::kPlayer2}) {
          if (static_cast<int>(playerPositions(env_loader, player, range.start_, range.end_).size()) >= num_positions) {
            example.eligible_players_ |= playerBit(player);
          }
        }
        if (example.eligible_players_ == 0) {
          ++bin_counts.too_few_positions_[range.phase_];
          continue;
        }
        examples.push_back(example);
        ++bin_counts.examples_[range.phase_];
      }
    }
  }

  std::vector<int> empty_bins;
  for (int bin = 0; bin < num_bins; ++bin) {
    const int low_elo = strength::nn_rank_min_elo + bin * strength::nn_rank_elo_interval;
    const PhaseCounts& bin_counts = counts[bin];
    std::cerr << "bin=" << bin << " elo=" << low_elo << "-" << (low_elo + strength::nn_rank_elo_interval - 1)
              << " games=" << bin_counts.games_;
    for (int phase = 0; phase < kNumPhases; ++phase) {
      std::cerr << " " << kPhaseNames[phase] << "=" << bin_counts.examples_[phase]
                << "(skipped_fewer_than_" << num_positions << "_positions=" << bin_counts.too_few_positions_[phase] << ")";
    }
    std::cerr << " skipped_missing_phase_tags=" << bin_counts.missing_tags_
              << " skipped_invalid_phase_tags=" << bin_counts.invalid_tags_ << std::endl;
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
  if (config::nn_type_name == "bt") { allocateBTGamePositions(); }
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

      if (strength::bt_use_phase_examples) {
        allocateBTPhaseExamplePositions(rank);
      } else if (strength::bt_use_same_game_per_rank) {
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
  if (strength::bt_use_phase_examples) {
    const size_t expected = static_cast<size_t>(strength::bt_num_batch_size) * strength::bt_num_rank_per_batch * strength::bt_num_position_per_rank;
    if (getSharedData()->bt_game_positions_.size() != expected) {
      throw std::runtime_error("phase example sampler produced " + std::to_string(getSharedData()->bt_game_positions_.size()) +
                               " positions, expected " + std::to_string(expected));
    }
  }
}

void StDataLoader::allocateBTPhaseExamplePositions(int rank)
{
  const std::vector<PhaseExample>& examples = getSharedData()->phase_examples_map_.at(rank);
  const PhaseExample& example = examples[Random::randInt() % examples.size()];
  const EnvironmentLoader& env_loader = getSharedData()->env_loaders_map_.at(rank)[example.env_id_];

  std::vector<env::Player> eligible_players;
  for (env::Player player : {env::Player::kPlayer1, env::Player::kPlayer2}) {
    if (example.eligible_players_ & playerBit(player)) { eligible_players.push_back(player); }
  }
  env::Player player = eligible_players[Random::randInt() % eligible_players.size()];

  std::vector<int> positions = playerPositions(env_loader, player, example.start_, example.end_);
  if (static_cast<int>(positions.size()) < strength::bt_num_position_per_rank) {
    throw std::runtime_error("phase example in bin " + std::to_string(rank) + " game " + std::to_string(example.env_id_) + " has too few positions");
  }
  std::random_shuffle(positions.begin(), positions.end());
  for (int j = 0; j < strength::bt_num_position_per_rank; ++j) {
    getSharedData()->bt_game_positions_.emplace_back(GamePosition(rank, example.env_id_, positions[j]));
  }
}

} // namespace strength
