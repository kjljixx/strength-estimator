#pragma once

#include "data_loader.h"
#include "st_configuration.h"
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace strength {

class GamePosition {
public:
  int rank_;
  int env_id_;
  int pos_;

  GamePosition() {}
  GamePosition(int rank, int env_id, int pos) : rank_(rank), env_id_(env_id), pos_(pos) {}

  inline bool operator==(const GamePosition& gp) const { return (rank_ == gp.rank_) && (env_id_ == gp.env_id_) && (pos_ == gp.pos_); }
  inline bool operator!=(const GamePosition& gp) const { return !(*this == gp); }
};

struct WinChainSlot {
  int game_id_ = -1;
  // 1 = white (kPlayer1 in chess), 2 = black (kPlayer2)
  int player_ = 1;
};

// One opening, midgame or endgame of a stored game: moves [start_, end_) of game phase_game_refs_[game_id_].
struct PhaseExample {
  int game_id_ = -1;
  int phase_ = 0;
  int start_ = 0;
  int end_ = 0;
  // bit 0: kPlayer1 has enough positions in the phase, bit 1: kPlayer2
  int eligible_players_ = 0;
};

// Where a complete game lives on disk; games are parsed only when a training step needs them.
struct PhaseGameRef {
  int file_index_ = 0;
  std::streamoff offset_ = 0;
};

struct PhaseBinStats {
  int games_ = 0;
  int missing_tags_ = 0;
  int invalid_tags_ = 0;
  int examples_[3] = {};
  int too_few_positions_[3] = {};
};

class StBatchDataPtr : public minizero::learner::BatchDataPtr {
public:
  float* rank_;
};

class StDataLoaderSharedData : public minizero::learner::DataLoaderSharedData {
public:
  void createDataPtr() override { data_ptr_ = std::make_shared<StBatchDataPtr>(); }
  inline std::shared_ptr<StBatchDataPtr> getDataPtr() { return std::static_pointer_cast<StBatchDataPtr>(data_ptr_); }

  std::vector<GamePosition> bt_game_positions_;
  std::map<int, int> rank_label_map_;
  std::map<int, std::vector<EnvironmentLoader>> env_loaders_map_;
  std::map<int, std::vector<PhaseExample>> phase_examples_map_;
  std::map<int, PhaseBinStats> phase_stats_;
  std::vector<std::string> phase_files_;
  std::vector<PhaseGameRef> phase_game_refs_;

  // Win-chain lazy loading: offset index + chains
  std::string chain_games_file_;
  std::vector<std::streamoff> chain_game_offsets_;
  std::vector<std::vector<WinChainSlot>> win_chains_;

  // Per-batch game pool (discarded each batch)
  std::vector<EnvironmentLoader> batch_chain_games_;
  std::unordered_map<int, int> batch_game_id_map_;
};

class StDataLoaderThread : public minizero::learner::DataLoaderThread {
public:
  StDataLoaderThread(int id, std::shared_ptr<minizero::utils::BaseSharedData> shared_data)
    : DataLoaderThread(id, shared_data) {}

protected:
  bool addEnvironmentLoader() override;
  bool sampleData() override;

  inline std::shared_ptr<StDataLoaderSharedData> getSharedData() { return std::static_pointer_cast<StDataLoaderSharedData>(shared_data_); }

private:
  void setAlphaZeroTrainingData(int batch_index);
  void setRankTrainingData(int batch_index);
  void setBTTrainingData(int batch_index);
  GamePosition sampleTrainingData();
  const EnvironmentLoader& getEnvLoader(const GamePosition& gp);
};

class StDataLoader : public minizero::learner::DataLoader {
public:
  StDataLoader(const std::string& conf_file_name);
  void initialize() override;
  void loadDataFromFile(const std::string& file_name) override;
  void loadWinChainsFromFile(const std::string& file_name);
  void sampleData() override;

  void createSharedData() override { shared_data_ = std::make_shared<StDataLoaderSharedData>(); }
  std::shared_ptr<minizero::utils::BaseSlaveThread> newSlaveThread(int id) override { return std::make_shared<StDataLoaderThread>(id, shared_data_); }
  inline std::shared_ptr<StDataLoaderSharedData> getSharedData() { return std::static_pointer_cast<StDataLoaderSharedData>(shared_data_); }

private:
  bool phase_examples_built_ = false;
  void allocateBTGamePositions();
  void allocateBTWinChainPositions();
  void indexPhaseGamesFile(const std::string& file_name);
  void finalizePhaseExamples();
  EnvironmentLoader loadPhaseGame(int game_id);
  void allocateBTPhaseExamplePositions();

  // Win-chain lazy loading helpers
  void indexChainGamesFile(const std::string& file_name);
  EnvironmentLoader loadChainGame(int game_id);
  void selectBTWinChains(std::vector<const std::vector<WinChainSlot>*>& selected_chains);
  void loadBTBatchGames(const std::vector<const std::vector<WinChainSlot>*>& selected_chains);
  void allocateBTWinChainPositionsFromLoaded(const std::vector<const std::vector<WinChainSlot>*>& selected_chains);
};

} // namespace strength
