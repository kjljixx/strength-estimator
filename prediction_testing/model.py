from __future__ import annotations

import math
from collections.abc import Callable, Sequence
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


class EloBaselineModel:
  def __init__(self, draw_rate: float = 0.1):
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
  def __init__(
    self,
    scorer: Callable[[str], float],
    sgf_loader: Callable[[str], str],
    *,
    across_games: str = "mean",
    color_normalization: bool = True,
    color_bias: float = 0.0,
    strength_scale: float = 1.0,
    draw_rate: float = 0.1,
    checkpoint_hash: str = "default",
  ):
    self.scorer = scorer
    self.sgf_loader = sgf_loader
    self.across_games = across_games
    self.color_normalization = color_normalization
    self.color_bias = color_bias
    self.strength_scale = strength_scale
    self.draw_rate = draw_rate
    self.checkpoint_hash = checkpoint_hash
    self._cache: dict[str, float] = {}

  def validate(self) -> None:
    if not 0 <= self.draw_rate < 1:
      raise ValueError("draw_rate must be in [0, 1)")

  def _score_game(self, game_id: str) -> float:
    key = f"{self.checkpoint_hash}:{game_id}"
    if key not in self._cache:
      self._cache[key] = self.scorer(self.sgf_loader(game_id))
    return self._cache[key]

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
      score = self._score_game(game.game_id)
      if self.color_normalization and game.target_color == "black":
        score = -score
      scores.append(score)
    return self._aggregate(scores)

  def predict(self, example: PredictionExample) -> GamePrediction:
    white_strength = self._player_strength(example, "white")
    black_strength = self._player_strength(example, "black")
    white, draw, black, predicted = strength_to_probs(
      white_strength,
      black_strength,
      draw_rate=self.draw_rate,
      strength_scale=self.strength_scale,
      color_bias=self.color_bias,
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
        "model": "strength_difference",
      },
    )

  def predict_batch(self, examples: Sequence[PredictionExample]) -> list[GamePrediction]:
    predictions: list[GamePrediction] = []
    for example in examples:
      predictions.append(self.predict(example))
    if len(predictions) != len(examples):
      raise RuntimeError("model failure removed examples")
    return predictions
