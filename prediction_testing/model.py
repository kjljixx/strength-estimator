from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Protocol

from prediction_testing.schemas import (
  GamePrediction,
  GameResult,
  PredictionExample,
)


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
  # 0 -0.110919
  # 1 -0.0404942
  # 2 0.0264197
  # 3 0.153585
  # 4 0.257167
  # 5 0.389452
  # 6 0.453127
  # 7 0.49278  
  score_to_elo_slope = 2099.249546736356
  score_to_elo_intercept = 1374.608727864828

  def __init__(
    self,
    scorer: Callable[[str], Mapping[str, float]],
    sgf_loader: Callable[[str], str],
    *,
    across_games: str = "mean",
    draw_rate: float = 0.04,
  ):
    self.scorer = scorer
    self.sgf_loader = sgf_loader
    self.across_games = across_games
    self.draw_rate = draw_rate

  def validate(self) -> None:
    if not 0 <= self.draw_rate < 1:
      raise ValueError("draw_rate must be in [0, 1)")

  def _score_game(self, game_id: str) -> Mapping[str, float]:
    return self.scorer(self.sgf_loader(game_id))

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

  def predict(self, example: PredictionExample) -> GamePrediction:
    white_strength = self._player_strength(example, "white")
    black_strength = self._player_strength(example, "black")
    white_elo = self.score_to_elo(white_strength)
    black_elo = self.score_to_elo(black_strength)
    white, draw, black, predicted = elo_to_probs(
      white_elo,
      black_elo,
      draw_rate=self.draw_rate,
    )
    return GamePrediction(
      example_id=example.example_id,
      white_win_probability=white,
      draw_probability=draw,
      black_win_probability=black,
      predicted_result=predicted,
      metadata={
        "white_strength": white_strength,
        "black_strength": black_strength,
        "white_estimated_elo": white_elo,
        "black_estimated_elo": black_elo,
        "model": "strength_difference",
      },
    )

  def predict_batch(self, examples: Sequence[PredictionExample]) -> list[GamePrediction]:
    return [self.predict(example) for example in examples]
