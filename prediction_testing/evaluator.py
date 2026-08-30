from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Sequence

from prediction_testing.model import GamePredictionModel
from prediction_testing.schemas import (
  EvaluationConfig,
  EvaluationResult,
  FilteredPredictionDataset,
  GamePrediction,
  GameResult,
  MetricReport,
  PredictionExample,
)


RESULT_INDEX = {
  GameResult.WHITE_WIN: 0,
  GameResult.DRAW: 1,
  GameResult.BLACK_WIN: 2,
}


def _actual_vector(result: GameResult) -> list[float]:
  return [
    1.0 if result == GameResult.WHITE_WIN else 0.0,
    1.0 if result == GameResult.DRAW else 0.0,
    1.0 if result == GameResult.BLACK_WIN else 0.0,
  ]


def _pred_vector(pred: GamePrediction) -> list[float]:
  return [
    pred.white_win_probability,
    pred.draw_probability,
    pred.black_win_probability,
  ]


def validate_probabilities(pred: GamePrediction) -> None:
  probs = _pred_vector(pred)
  if any(p < 0 or p > 1 for p in probs):
    raise ValueError(f"{pred.example_id}: probability out of range")
  if abs(sum(probs) - 1.0) > 1e-5:
    raise ValueError(f"{pred.example_id}: probabilities do not sum to 1")


def accuracy(examples: Sequence[PredictionExample], preds: Sequence[GamePrediction]) -> float:
  correct = 0
  for example, pred in zip(examples, preds):
    if pred.predicted_result == example.prediction_game.result:
      correct += 1
  return correct / len(examples)


def log_loss(examples: Sequence[PredictionExample], preds: Sequence[GamePrediction]) -> float:
  total = 0.0
  for example, pred in zip(examples, preds):
    probs = _pred_vector(pred)
    idx = RESULT_INDEX[example.prediction_game.result]
    total -= math.log(max(probs[idx], 1e-15))
  return total / len(examples)


def brier_score(examples: Sequence[PredictionExample], preds: Sequence[GamePrediction]) -> float:
  total = 0.0
  for example, pred in zip(examples, preds):
    actual = _actual_vector(example.prediction_game.result)
    predicted = _pred_vector(pred)
    total += sum((a - p) ** 2 for a, p in zip(actual, predicted))
  return total / len(examples)


class PredictionEvaluator:
  def run(
    self,
    model: GamePredictionModel,
    dataset: FilteredPredictionDataset,
    config: EvaluationConfig,
    output_dir: Path,
  ) -> EvaluationResult:
    model.validate()
    predictions = model.predict_batch(list(dataset.examples))
    report = self.score(dataset.examples, predictions, config)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
      "n_examples": report.n_examples,
      "accuracy": report.accuracy,
      "draw_rate": report.draw_rate,
      "log_loss": report.log_loss,
      "brier_score": report.brier_score,
      "slices": report.slices,
      "sample_predictions": [{
          "example_id": pred.example_id,
          "predicted_result": pred.predicted_result.name,
          "white_win_probability": pred.white_win_probability,
          "draw_probability": pred.draw_probability,
          "black_win_probability": pred.black_win_probability,
        } for pred in predictions[:10]],
    }
    (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return EvaluationResult(
      model_name=type(model).__name__,
      metric_report=report,
      predictions=tuple(predictions),
    )

  def score(
    self,
    examples: Sequence[PredictionExample],
    predictions: Sequence[GamePrediction],
    config: EvaluationConfig,
  ) -> MetricReport:
    by_id = {pred.example_id: pred for pred in predictions}
    if len(by_id) != len(predictions):
      raise ValueError("duplicate predictions")
    ordered: list[GamePrediction] = []
    for example in examples:
      if example.example_id not in by_id:
        raise ValueError(f"missing prediction for {example.example_id}")
      pred = by_id.pop(example.example_id)
      validate_probabilities(pred)
      ordered.append(pred)
    if by_id:
      raise ValueError(f"unknown predictions: {sorted(by_id)}")

    acc = accuracy(examples, ordered)
    draw_rate = sum(1 for ex in examples if ex.prediction_game.result == GameResult.DRAW) / len(examples)
    ll = log_loss(examples, ordered)
    brier = brier_score(examples, ordered)

    slices: dict[str, dict[str, tuple[float, int]]] = {}
    if config.report_by_elo:
      slices["elo_bucket"] = self._slice_metric(examples, ordered, config, _elo_bucket)
    if config.report_by_elo_difference:
      slices["elo_diff_bucket"] = self._slice_metric(examples, ordered, config, _elo_diff_bucket)
    if config.report_by_month:
      slices["month"] = self._slice_metric(examples, ordered, config, _month_bucket)
    if config.report_by_context_size:
      slices["context_size"] = self._slice_metric(
        examples, ordered, config,
        lambda ex: str(len(ex.white_context.games)),
      )

    return MetricReport(
      n_examples=len(examples),
      accuracy=acc,
      draw_rate = draw_rate,
      log_loss=ll,
      brier_score=brier,
      slices=slices,
    )

  def _slice_metric(
    self,
    examples: Sequence[PredictionExample],
    predictions: Sequence[GamePrediction],
    config: EvaluationConfig,
    key_fn,
  ) -> dict[str, tuple[float, int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, example in enumerate(examples):
      groups[key_fn(example)].append(idx)
    out: dict[str, tuple[float, int]] = {}
    for key, indices in sorted(groups.items(), key=lambda x: (float(x[0].split("-")[0]), float(x[0].split("-")[1])) if "-" in x[0] else x):
      if len(indices) < config.minimum_slice_size:
        continue
      sub_examples = [examples[i] for i in indices]
      sub_preds = [predictions[i] for i in indices]
      out[key] = (accuracy(sub_examples, sub_preds), len(sub_examples))
    return out


def _elo_bucket(example: PredictionExample) -> str:
  avg = (example.prediction_game.white_elo + example.prediction_game.black_elo) // 2
  return f"{avg // 200 * 200}-{(avg // 200 + 1) * 200}"


def _elo_diff_bucket(example: PredictionExample) -> str:
  diff = abs(example.prediction_game.white_elo - example.prediction_game.black_elo)
  return f"{diff // 30 * 30}-{(diff // 30 + 1) * 30}"


def _month_bucket(example: PredictionExample) -> str:
  dt: datetime = example.prediction_game.played_at
  return f"{dt.year}-{dt.month:02d}"
