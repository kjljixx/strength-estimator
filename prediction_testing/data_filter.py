from __future__ import annotations

import hashlib
import mmap
import re
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Protocol, Sequence

from tqdm import tqdm

from prediction_testing.schemas import (
  ContextGame,
  ContextPolicy,
  ExampleManifest,
  FilteredPredictionDataset,
  GameRecord,
  GameResult,
  PlayerContext,
  PredictionExample,
)

SGF_HEADER = re.compile(
  rb"\(;GM\[chess\]RE\[(-?\d+(?:\.\d+)?)\]EV\[([^\]]*)\]"
  rb"PW\[([^\]]*)\]PB\[([^\]]*)\]DT\[(\d{4}\.\d{2}\.\d{2})\]"
  rb"(?:WR\[(\d+)\])?(?:BR\[(\d+)\])?"
)
EXCLUSION_KEYS = (
  "unsupported_event",
  "invalid_result",
  "missing_player",
  "missing_rating",
  "rating_out_of_range",
  "duplicate_game",
  "white_insufficient_context",
  "black_insufficient_context",
  "context_too_old",
  "ambiguous_same_day_order",
  "max_predictions_per_player",
)


def parse_result(raw: str | bytes) -> GameResult | None:
  value = float(raw)
  if value > 0:
    return GameResult.WHITE_WIN
  if value < 0:
    return GameResult.BLACK_WIN
  return GameResult.DRAW


def game_filter(game: GameRecord, policy: ContextPolicy) -> str | None:
  if game.event not in policy.allowed_events:
    return "unsupported_event"
  if policy.require_both_ratings and (game.white_elo <= 0 or game.black_elo <= 0):
    return "missing_rating"
  if (
    game.white_elo < policy.min_elo or game.white_elo > policy.max_elo
    or game.black_elo < policy.min_elo or game.black_elo > policy.max_elo
  ):
    return "rating_out_of_range"
  return None


def target_perspective(
  game: GameRecord,
  player_id: str,
) -> ContextGame | None:
  if player_id == game.white_player:
    color: str = "white"
    target_elo = game.white_elo
    opponent_elo = game.black_elo
    if game.result == GameResult.WHITE_WIN:
      outcome = "win"
    elif game.result == GameResult.DRAW:
      outcome = "draw"
    else:
      outcome = "loss"
  elif player_id == game.black_player:
    color = "black"
    target_elo = game.black_elo
    opponent_elo = game.white_elo
    if game.result == GameResult.BLACK_WIN:
      outcome = "win"
    elif game.result == GameResult.DRAW:
      outcome = "draw"
    else:
      outcome = "loss"
  else:
    warnings.warn(
      f"player {player_id!r} not in game {game.game_id!r}; skipping context",
      stacklevel=2,
    )
    return None
  return ContextGame(
    game_id=game.game_id,
    target_color=color,
    target_elo_at_game=target_elo,
    opponent_elo_at_game=opponent_elo,
    result_from_target_perspective=outcome,
    played_at=game.played_at,
  )


class ContextSelector(Protocol):
  def select(
    self,
    eligible_games: Sequence[ContextGame],
    *,
    count: int,
    prediction_game: GameRecord,
    seed: int,
  ) -> tuple[ContextGame, ...]: ...


class MostRecentSelector:
  def select(
    self,
    eligible_games: Sequence[ContextGame],
    *,
    count: int,
    prediction_game: GameRecord,
    seed: int,
  ) -> tuple[ContextGame, ...]:
    ordered = sorted(eligible_games, key=lambda g: (g.played_at, g.game_id))
    return tuple(ordered[-count:])


def make_selector() -> ContextSelector:
  return MostRecentSelector()


class GameCatalog:
  def __init__(self) -> None:
    self._games: list[GameRecord] = []
    self._sgf_spans: dict[str, tuple[Path, int, int]] = {}
    self._inline_sgf: dict[str, str] = {}
    self._sgf_cache: dict[str, str] = {}

  def add(self, game: GameRecord, sgf: str) -> None:
    self._games.append(game)
    self._inline_sgf[game.game_id] = sgf

  def index_paths(self, paths: Sequence[Path]) -> None:
    for path in tqdm(paths, desc="Loading catalog", unit="file"):
      if path.stat().st_size == 0:
        continue
      with path.open("rb") as handle, mmap.mmap(
        handle.fileno(), 0, access=mmap.ACCESS_READ,
      ) as content:
        with tqdm(
          desc=f"Parsing {path.name}",
          unit="game",
          leave=False,
          position=1,
        ) as progress:
          for idx, match in enumerate(SGF_HEADER.finditer(content)):
            result = parse_result(match.group(1))
            if result is not None:
              played_at = datetime.strptime(match.group(5).decode(), "%Y.%m.%d")
              white_elo = int(match.group(6) or 0)
              black_elo = int(match.group(7) or 0)
              digest = hashlib.sha1(match.group(0)).hexdigest()[:12]
              game_id = f"{path.stem}:{digest}:{idx}"
              self._games.append(
                GameRecord(
                  game_id=game_id,
                  played_at=played_at,
                  white_player=match.group(3).decode("utf-8", errors="replace"),
                  black_player=match.group(4).decode("utf-8", errors="replace"),
                  white_elo=white_elo,
                  black_elo=black_elo,
                  result=result,
                  event=match.group(2).decode("utf-8", errors="replace") or "Blitz",
                )
              )
              self._sgf_spans[game_id] = (path, match.start(), match.end())
            progress.update()

  @property
  def games(self) -> tuple[GameRecord, ...]:
    return tuple(sorted(self._games, key=lambda g: (g.played_at, g.game_id)))

  def load_sgf(self, game_id: str) -> str:
    cached = self._sgf_cache.get(game_id)
    if cached is not None:
      return cached
    inline = self._inline_sgf.get(game_id)
    if inline is not None:
      self._sgf_cache[game_id] = inline
      return inline
    path, start, end = self._sgf_spans[game_id]
    with path.open("rb") as handle:
      handle.seek(start)
      sgf = handle.read(end - start).decode("utf-8", errors="replace")
    self._sgf_cache[game_id] = sgf
    return sgf


class PredictionDataFilter:
  def __init__(self, selector: ContextSelector | None = None):
    self._selector = selector or make_selector()

  def load_catalog(self, paths: Sequence[Path]) -> GameCatalog:
    catalog = GameCatalog()
    catalog.index_paths(paths)
    return catalog

  def build_examples(
    self,
    catalog: GameCatalog,
    policy: ContextPolicy,
  ) -> FilteredPredictionDataset:
    games = catalog.games
    selector = self._selector
    exclusions: Counter[str] = Counter({k: 0 for k in EXCLUSION_KEYS})
    seen_ids: set[str] = set()
    player_predictions: Counter[str] = Counter()
    examples: list[PredictionExample] = []
    manifests: list[ExampleManifest] = []

    by_player: dict[str, list[GameRecord]] = defaultdict(list)
    for game in tqdm(games, desc="Indexing players", unit="game"):
      by_player[game.white_player].append(game)
      by_player[game.black_player].append(game)

    for game in tqdm(games, desc="Creating examples", unit="game"):
      if game.game_id in seen_ids:
        exclusions["duplicate_game"] += 1
        continue
      seen_ids.add(game.game_id)

      reason = game_filter(game, policy)
      if reason is not None:
        exclusions[reason] += 1
        continue
      if policy.max_prediction_games_per_player is not None:
        if (
          player_predictions[game.white_player] >= policy.max_prediction_games_per_player
          or player_predictions[game.black_player] >= policy.max_prediction_games_per_player
        ):
          exclusions["max_predictions_per_player"] += 1
          continue

      white_eligible = self._eligible_context(
        by_player[game.white_player], game, game.white_player, policy,
      )
      black_eligible = self._eligible_context(
        by_player[game.black_player], game, game.black_player, policy,
      )


      if len(white_eligible) < policy.context_size:
        exclusions["white_insufficient_context"] += 1
        continue
      if len(black_eligible) < policy.context_size:
        exclusions["black_insufficient_context"] += 1
        continue

      white_ctx = selector.select(
        white_eligible, count=policy.context_size, prediction_game=game, seed=policy.seed,
      )
      black_ctx = selector.select(
        black_eligible, count=policy.context_size, prediction_game=game, seed=policy.seed + 1,
      )

      example_id = f"{game.game_id}@{game.played_at.isoformat()}"
      example = PredictionExample(
        example_id=example_id,
        prediction_game=game,
        white_context=PlayerContext(game.white_player, white_ctx, game.played_at),
        black_context=PlayerContext(game.black_player, black_ctx, game.played_at),
      )
      examples.append(example)
      manifests.append(
        ExampleManifest(
          example_id=example_id,
          prediction_game_id=game.game_id,
          white_context_ids=tuple(g.game_id for g in white_ctx),
          black_context_ids=tuple(g.game_id for g in black_ctx),
        )
      )
      player_predictions[game.white_player] += 1
      player_predictions[game.black_player] += 1

    return FilteredPredictionDataset(
      catalog=catalog,
      examples=tuple(examples),
      manifests=tuple(manifests),
      exclusion_counts=dict(exclusions),
    )

  def _eligible_context(
    self,
    player_games: Sequence[GameRecord],
    prediction_game: GameRecord,
    player_id: str,
    policy: ContextPolicy,
  ) -> list[ContextGame]:
    eligible: list[ContextGame] = []
    min_age = timedelta(seconds=policy.min_context_age_seconds)
    max_age = (
      timedelta(days=policy.max_context_age_days)
      if policy.max_context_age_days is not None
      else None
    )
    for game in player_games:
      if game.game_id == prediction_game.game_id:
        continue
      if game.played_at >= prediction_game.played_at:
        continue
      if prediction_game.played_at - game.played_at < min_age:
        continue
      if policy.exclude_same_day_context and game.played_at.date() >= prediction_game.played_at.date():
        continue
      if max_age is not None and prediction_game.played_at - game.played_at > max_age:
        continue
      if game_filter(game, policy) is not None:
        continue
      ctx = target_perspective(game, player_id)
      if ctx is not None:
        eligible.append(ctx)
    return eligible
