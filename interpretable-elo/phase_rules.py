"""Shared game-phase rules: opening, midgame and endgame boundaries."""

from __future__ import annotations

import chess

PHASES = ("opening", "midgame", "endgame")


def mixedness(board: chess.Board) -> int:
  def score(rank: int, white: int, black: int) -> int:
    if white == 0:
      return {1: 1 + rank, 2: max(0, 8 - rank) if rank < 6 else 0,
              3: 10 - rank if rank < 7 else 0,
              4: 10 - rank if rank < 7 else 0}.get(black, 0)
    if white == 1:
      return {0: 9 - rank, 1: 5 + abs(4 - rank), 2: 11 - rank,
              3: 12 - rank}.get(black, 0)
    if white == 2:
      return {0: rank if rank > 2 else 0, 1: rank + 3,
              2: 7}.get(black, 0)
    if white == 3:
      return {0: rank + 2 if rank > 1 else 0,
              1: rank + 4}.get(black, 0)
    if white == 4 and black == 0:
      return rank + 2 if rank > 1 else 0
    return 0

  total = 0
  for rank in range(1, 8):
    for file in range(7):
      region = sum(1 << chess.square(file + x, rank - 1 + y)
                   for x in range(2) for y in range(2))
      total += score(rank, chess.popcount(board.occupied_co[chess.WHITE] & region),
                     chess.popcount(board.occupied_co[chess.BLACK] & region))
  return total


def phase_boundaries(boards: list[chess.Board]) -> tuple[int | None, int | None]:
  def piece_count(board: chess.Board) -> int:
    return chess.popcount(board.occupied & ~(board.kings | board.pawns))

  middle = next((index for index, board in enumerate(boards)
                 if piece_count(board) <= 10
                 or chess.popcount(board.occupied_co[chess.WHITE] & chess.BB_RANK_1) < 4
                 or chess.popcount(board.occupied_co[chess.BLACK] & chess.BB_RANK_8) < 4
                 or mixedness(board) > 150), None)
  end = next((index for index, board in enumerate(boards)
              if piece_count(board) <= 6), None) if middle is not None else None
  if middle == end:
    middle = None
  return middle, end


def phase_for(ply_index: int, middle: int | None, end: int | None) -> str:
  if end is not None and ply_index >= end:
    return "endgame"
  if middle is not None and ply_index >= middle:
    return "midgame"
  return "opening"


def engine_move_to_move(board: chess.Board, uci: str) -> chess.Move:
  move = chess.Move.from_uci(uci)
  moving_piece = board.piece_at(move.from_square)
  target_piece = board.piece_at(move.to_square)
  if (moving_piece is not None and moving_piece.piece_type == chess.KING
      and target_piece is not None and target_piece.piece_type == chess.ROOK
      and target_piece.color == moving_piece.color):
    file = chess.G1 if chess.square_file(move.to_square) > chess.square_file(move.from_square) else chess.C1
    return chess.Move(move.from_square, file + 56 * chess.square_rank(move.from_square) // 7)
  reaches_last_rank = chess.square_rank(move.to_square) in (0, 7)
  if moving_piece is not None and moving_piece.piece_type == chess.PAWN and reaches_last_rank and move.promotion is None:
    return chess.Move(move.from_square, move.to_square, chess.KNIGHT)
  return move


def boards_after_moves(engine_moves: list[str]) -> list[chess.Board]:
  board = chess.Board()
  boards: list[chess.Board] = []
  for uci in engine_moves:
    move = engine_move_to_move(board, uci)
    if not board.is_legal(move):
      raise ValueError(f"illegal move {uci} in {board.fen()}")
    board.push(move)
    boards.append(board.copy(stack=False))
  return boards


def phase_ranges(move_count: int, middle: int | None, end: int | None) -> dict[str, tuple[int, int]]:
  starts = {"opening": 0, "midgame": middle, "endgame": end}
  ordered = [(phase, start) for phase, start in starts.items() if start is not None]
  ranges = {}
  for position, (phase, start) in enumerate(ordered):
    stop = ordered[position + 1][1] if position + 1 < len(ordered) else move_count
    if stop > start:
      ranges[phase] = (start, stop)
  return ranges
