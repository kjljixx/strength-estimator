from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Protocol

from tqdm import tqdm

from prediction_testing.schemas import (
  GamePrediction,
  GameResult,
  ProgressivePrediction,
  PredictionExample,
)
from prediction_testing.sgf import game_prefix, keep_last_moves, move_count


class GamePredictionModel(Protocol):
  def validate(self) -> None: ...

  def predict(self, example: PredictionExample) -> GamePrediction: ...

  def predict_batch(
    self,
    examples: Sequence[PredictionExample],
  ) -> list[GamePrediction]: ...


def strength_to_probs(
  white_strength: float,
  black_strength: float,
  *,
  draw_rate: float,
  strength_scale: float = 1.0,
  color_bias: float = 0.0,
) -> tuple[float, float, float, GameResult]:
  diff = strength_scale * (white_strength - black_strength) + color_bias
  p_white = 1.0 / (1.0 + math.exp(-diff))
  p_black = 1.0 / (1.0 + math.exp(diff))
  remaining = 1.0 - draw_rate
  white = remaining * p_white
  black = remaining * p_black
  probs = {
    GameResult.WHITE_WIN: white,
    GameResult.DRAW: draw_rate,
    GameResult.BLACK_WIN: black,
  }
  return white, draw_rate, black, max(probs, key=probs.get)


def elo_to_probs(
  white_elo: float,
  black_elo: float,
  *,
  draw_rate: float,
  color_bias: float = 0.0,
) -> tuple[float, float, float, GameResult]:
  expected_white = 1.0 / (1.0 + 10.0 ** ((black_elo - white_elo - color_bias) / 400.0))
  white = (1.0 - draw_rate) * expected_white
  black = (1.0 - draw_rate) * (1.0 - expected_white)
  probs = {
    GameResult.WHITE_WIN: white,
    GameResult.DRAW: draw_rate,
    GameResult.BLACK_WIN: black,
  }
  return white, draw_rate, black, max(probs, key=probs.get)


class EloBaselineModel:
  def __init__(self, draw_rate: float = 0.04):
    self.draw_rate = draw_rate

  def validate(self) -> None:
    if not 0 <= self.draw_rate < 1:
      raise ValueError("draw_rate must be in [0, 1)")

  def predict(self, example: PredictionExample) -> GamePrediction:
    diff = example.prediction_game.white_elo - example.prediction_game.black_elo
    white, draw, black, predicted = strength_to_probs(
      diff / 400.0,
      0.0,
      draw_rate=self.draw_rate,
    )
    return GamePrediction(
      example_id=example.example_id,
      white_win_probability=white,
      draw_probability=draw,
      black_win_probability=black,
      predicted_result=predicted,
      metadata={"model": "elo_baseline"},
    )

  def predict_batch(self, examples: Sequence[PredictionExample]) -> list[GamePrediction]:
    return [self.predict(example) for example in examples]


class StrengthDifferenceModel:
  default_score_to_elo_slope = 2099.249546736356
  default_score_to_elo_intercept = 1374.608727864828
  legacy_score_to_elo_slope = 357.702991
  legacy_score_to_elo_intercept = 1941.294157

  def __init__(
    self,
    scorer: Callable[[str], Mapping[str, float]],
    sgf_loader: Callable[[str], str],
    *,
    across_games: str = "mean",
    draw_rate: float = 0.04,
    context_last_n_moves: int | None = None,
    legacy: bool = False,
  ):
    self.scorer = scorer
    self.sgf_loader = sgf_loader
    self.across_games = across_games
    self.draw_rate = draw_rate
    self.context_last_n_moves = context_last_n_moves
    self.legacy = legacy
    (
      self.score_to_elo_slope,
      self.score_to_elo_intercept,
    ) = (
      (self.legacy_score_to_elo_slope, self.legacy_score_to_elo_intercept)
      if legacy
      else (self.default_score_to_elo_slope, self.default_score_to_elo_intercept)
    )
    self._game_score_cache: dict[str, Mapping[str, float]] = {}

  def validate(self) -> None:
    if not 0 <= self.draw_rate < 1:
      raise ValueError("draw_rate must be in [0, 1)")
    if self.context_last_n_moves is not None and self.context_last_n_moves <= 0:
      raise ValueError("context_last_n_moves must be positive")

  def _score_game(self, game_id: str) -> Mapping[str, float]:
    if game_id in self._game_score_cache:
      return self._game_score_cache[game_id]
    sgf = self.sgf_loader(game_id)
    if self.context_last_n_moves is not None:
      sgf = keep_last_moves(sgf, self.context_last_n_moves)
    scores = self.scorer(sgf)
    self._game_score_cache[game_id] = scores
    return scores

  def _aggregate(self, values: Sequence[float]) -> float:
    if not values:
      return 0.0
    if self.across_games == "median":
      ordered = sorted(values)
      mid = len(ordered) // 2
      return ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
    if self.across_games == "recency_weighted":
      total = sum((i + 1) * v for i, v in enumerate(values))
      weights = sum(i + 1 for i in range(len(values)))
      return total / weights
    return sum(values) / len(values)

  def _player_strength(self, example: PredictionExample, side: str) -> float:
    ctx = example.white_context if side == "white" else example.black_context
    scores: list[float] = []
    for game in ctx.games:
      scores.append(self._score_game(game.game_id)[game.target_color])
    return self._aggregate(scores)

  def score_to_elo(self, score: float) -> float:
    return self.score_to_elo_slope * score + self.score_to_elo_intercept

  def _prediction_from_strengths(
    self,
    example: PredictionExample,
    white_strength: float,
    black_strength: float,
    metadata: Mapping[str, object] | None = None,
  ) -> GamePrediction:
    white_elo = self.score_to_elo(white_strength)
    black_elo = self.score_to_elo(black_strength)
    white, draw, black, predicted = elo_to_probs(
      white_elo,
      black_elo,
      draw_rate=self.draw_rate,
    )
    details = {
      "white_strength": white_strength,
      "black_strength": black_strength,
      "white_estimated_elo": white_elo,
      "black_estimated_elo": black_elo,
      "model": "strength_difference",
      "legacy_calibration": self.legacy,
    }
    if metadata:
      details.update(metadata)
    return GamePrediction(
      example_id=example.example_id,
      white_win_probability=white,
      draw_probability=draw,
      black_win_probability=black,
      predicted_result=predicted,
      metadata=details,
    )

  def predict(self, example: PredictionExample) -> GamePrediction:
    white_strength = self._player_strength(example, "white")
    black_strength = self._player_strength(example, "black")
    return self._prediction_from_strengths(example, white_strength, black_strength)

  def predict_progressive(
    self,
    example: PredictionExample,
    *,
    typical_moves_per_player_per_game: float,
    current_move_weight: float,
  ) -> list[ProgressivePrediction]:
    if typical_moves_per_player_per_game <= 0:
      raise ValueError("typical_moves_per_player_per_game must be positive")
    if not math.isfinite(current_move_weight) or current_move_weight < 0:
      raise ValueError("current_move_weight must be finite and non-negative")

    context = {
      "white": self._player_strength(example, "white"),
      "black": self._player_strength(example, "black"),
    }
    sgf = self.sgf_loader(example.prediction_game.game_id)
    total_plies = move_count(sgf)
    baseline = self._prediction_from_strengths(
      example,
      context["white"],
      context["black"],
      {"observed_plies": 0, "current_game_weight": 0.0},
    )
    trajectory = [ProgressivePrediction(
      example_id=example.example_id,
      observed_plies=0,
      is_recorded_endpoint=total_plies == 0,
      prediction=baseline,
    )]

    context_counts = {
      "white": len(example.white_context.games),
      "black": len(example.black_context.games),
    }
    for plies in range(2, total_plies + 1):
      current = self.scorer(game_prefix(sgf, plies))
      observation_counts = {"white": (plies + 1) // 2, "black": plies // 2}
      blended: dict[str, float] = {}
      fractions: dict[str, float] = {}
      for side in ("white", "black"):
        historical_weight = (
          context_counts[side] * typical_moves_per_player_per_game
        )
        live_weight = current_move_weight * observation_counts[side]
        total_weight = historical_weight + live_weight
        blended[side] = (
          historical_weight * context[side] + live_weight * current[side]
        ) / total_weight if total_weight else context[side]
        fractions[side] = live_weight / total_weight if total_weight else 0.0
      prediction = self._prediction_from_strengths(
        example,
        blended["white"],
        blended["black"],
        {
          "observed_plies": plies,
          "white_context_strength": context["white"],
          "black_context_strength": context["black"],
          "white_current_strength": current["white"],
          "black_current_strength": current["black"],
          "white_context_game_count": context_counts["white"],
          "black_context_game_count": context_counts["black"],
          "white_current_observation_count": observation_counts["white"],
          "black_current_observation_count": observation_counts["black"],
          "white_current_game_fraction": fractions["white"],
          "black_current_game_fraction": fractions["black"],
        },
      )
      trajectory.append(ProgressivePrediction(
        example_id=example.example_id,
        observed_plies=plies,
        is_recorded_endpoint=plies == total_plies,
        prediction=prediction,
      ))
    return trajectory

  def predict_batch(self, examples: Sequence[PredictionExample]) -> list[GamePrediction]:
    return [
      self.predict(example)
      for example in tqdm(examples, desc="Scoring predictions", unit="game")
    ]
