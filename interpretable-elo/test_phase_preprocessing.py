from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

import chess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "interpretable-elo"))

from phase_rules import boards_after_moves, engine_move_to_move, phase_boundaries, phase_ranges

SPEC = importlib.util.spec_from_file_location("sgf_filter", ROOT / "scripts" / "sgf_filter_random_sample.py")
sgf_filter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sgf_filter)

ARGS = sgf_filter.build_parser().parse_args(["--elo-interval", "100"])


def sgf_line(white_elo, black_elo, moves=("e2e4", "e7e5")):
  body = "".join(f";{'B' if i % 2 == 0 else 'W'}[{move}]TM[300]" for i, move in enumerate(moves))
  return f"(;GM[chess]RE[1.0]PW[a]PB[b]DT[2024.01.01]WR[{white_elo}]BR[{black_elo}]{body})\n"


def random_game_moves(seed, plies=400):
  rng = random.Random(seed)
  board = chess.Board()
  moves = []
  while len(moves) < plies and not board.is_game_over():
    move = rng.choice(list(board.legal_moves))
    moves.append(sgf_filter_engine_uci(board, move))
    board.push(move)
  return moves


def sgf_filter_engine_uci(board, move):
  if not board.is_castling(move):
    return move.uci()
  rook_square = chess.H1 if board.is_kingside_castling(move) else chess.A1
  if board.turn == chess.BLACK:
    rook_square += 56
  return chess.square_name(move.from_square) + chess.square_name(rook_square)


def test_elo_bin_boundaries():
  classify = lambda white, black: sgf_filter.classify(sgf_line(white, black), ARGS, set())
  assert classify(1000, 1000) == (0, None)
  assert classify(1099, 1000) == (0, None)
  assert classify(1099, 1100) == (None, "players_in_different_bins")
  assert classify(1100, 1100) == (1, None)
  assert classify(2599, 2500) == (15, None)
  assert classify(999, 1000) == (None, "rating_out_of_range")
  assert classify(2600, 2599) == (None, "rating_out_of_range")


def test_held_out_games_are_excluded():
  line = sgf_line(1500, 1500)
  held_out = {sgf_filter.fingerprint(line)}
  assert sgf_filter.classify(line, ARGS, held_out) == (None, "held_out_game")
  assert sgf_filter.classify(sgf_line(1500, 1500, ("d2d4", "d7d5")), ARGS, held_out)[0] == 5


def test_castling_board_reconstruction():
  boards = boards_after_moves("e2e4 e7e5 g1f3 b8c6 f1c4 g8f6 e1h1 f8c5 d2d3 e8h8".split())
  assert boards[-1].fen() == "r1bq1rk1/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQ1RK1 w - - 1 6"


def test_phase_ranges_partition_the_game():
  assert phase_ranges(10, 4, 8) == {"opening": (0, 4), "midgame": (4, 8), "endgame": (8, 10)}
  assert phase_ranges(10, None, None) == {"opening": (0, 10)}
  assert phase_ranges(10, 4, None) == {"opening": (0, 4), "midgame": (4, 10)}


def test_transition_positions_match_phase_boundaries():
  moves, middle, end = next(
    (moves, *phase_boundaries(boards_after_moves(moves)))
    for moves in map(random_game_moves, range(50))
    if phase_boundaries(boards_after_moves(moves))[1] is not None)
  boards = boards_after_moves(moves)
  assert 0 < middle < end < len(moves)
  assert phase_boundaries(boards[:middle]) == (None, None)
  assert phase_boundaries(boards[:middle + 1])[0] == middle
  assert phase_boundaries(boards[:end])[1] is None
  assert phase_boundaries(boards[:end + 1])[1] == end
  ranges = phase_ranges(len(moves), middle, end)
  assert ranges == {"opening": (0, middle), "midgame": (middle, end), "endgame": (end, len(moves))}


def test_phase_tags_added_before_first_move():
  moves = next(moves for moves in map(random_game_moves, range(50))
               if phase_boundaries(boards_after_moves(moves))[1] is not None)
  middle, end = phase_boundaries(boards_after_moves(moves))
  annotated, reason, phases = sgf_filter.add_phase_tags(sgf_line(1500, 1500, moves))
  assert reason is None
  assert f"BR[1500]PM[{middle}]PE[{end}];B[" in annotated
  assert phases == ("opening", "midgame", "endgame")


def test_missing_phases_are_tagged_minus_one():
  annotated, reason, phases = sgf_filter.add_phase_tags(sgf_line(1500, 1500, ("e2e4", "e7e5", "g1f3", "b8c6")))
  assert reason is None
  assert "PM[-1]PE[-1]" in annotated
  assert phases == ("opening",)


def test_illegal_moves_are_skipped_with_reason():
  assert sgf_filter.add_phase_tags(sgf_line(1500, 1500, ("e2e5",)))[1] == "illegal_move_replay"


def test_suffixless_last_rank_pawn_move_is_knight_promotion():
  board = chess.Board("8/3P2K1/4k3/8/6P1/8/8/8 w - - 1 55")
  move = engine_move_to_move(board, "d7d8")
  assert move.promotion == chess.KNIGHT and board.is_legal(move)
  assert engine_move_to_move(board, "d7d8q").promotion == chess.QUEEN
