"""Encoding round-trip and plane-layout tests on 100+ positions."""

from __future__ import annotations

import random

import chess
import numpy as np

from chess_rl.encoding import (
    INPUT_PLANES,
    NUM_ACTIONS,
    PIECE_TO_PLANE,
    decode_action,
    encode_board,
    encode_move,
    legal_action_mask,
)


def test_starting_board_planes() -> None:
    board = chess.Board()
    planes = encode_board(board)
    assert planes.shape == (INPUT_PLANES, 8, 8)
    assert planes[PIECE_TO_PLANE[(chess.PAWN, chess.WHITE)], 1, :].sum() == 8
    assert planes[PIECE_TO_PLANE[(chess.PAWN, chess.BLACK)], 6, :].sum() == 8
    assert planes[PIECE_TO_PLANE[(chess.KING, chess.WHITE)], 0, 4] == 1
    assert planes[PIECE_TO_PLANE[(chess.KING, chess.BLACK)], 7, 4] == 1
    for plane in range(12, 16):
        assert planes[plane].sum() == 64
    assert planes[16].sum() == 64
    assert planes[17].sum() == 0


def test_no_castling_rights() -> None:
    board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1")
    planes = encode_board(board)
    for plane in range(12, 16):
        assert planes[plane].sum() == 0


def test_en_passant_target() -> None:
    board = chess.Board("rnbqkbnr/pp1ppppp/8/2pP4/8/8/PPP1PPPP/RNBQKBNR w KQkq c6 0 3")
    planes = encode_board(board)
    assert planes[17, 5, 2] == 1.0
    assert planes[17].sum() == 1.0


def test_black_to_move_plane_off() -> None:
    board = chess.Board()
    board.push_uci("e2e4")
    planes = encode_board(board)
    assert planes[16].sum() == 0


def test_move_round_trip_starting_position() -> None:
    board = chess.Board()
    legal = list(board.legal_moves)
    assert len(legal) == 20
    for move in legal:
        action = encode_move(move, board)
        assert 0 <= action < NUM_ACTIONS
        decoded = decode_action(action, board)
        assert decoded == move, f"{move.uci()} round-trip → {decoded}"


def test_legal_action_mask_starting() -> None:
    board = chess.Board()
    mask = legal_action_mask(board)
    assert mask.dtype == np.bool_
    assert mask.sum() == 20


def test_round_trip_random_walk_120_positions() -> None:
    rng = random.Random(42)
    board = chess.Board()
    positions_checked = 0
    while positions_checked < 120:
        legal = list(board.legal_moves)
        if not legal:
            board = chess.Board()
            continue
        for move in legal:
            action = encode_move(move, board)
            decoded = decode_action(action, board)
            assert decoded == move, (
                f"Round-trip failed at {board.fen()}: {move.uci()} → {decoded}"
            )
        board.push(rng.choice(legal))
        positions_checked += 1
        if board.is_game_over():
            board = chess.Board()


def test_all_four_promotion_types_round_trip() -> None:
    board = chess.Board("8/P7/8/8/8/8/8/k1K5 w - - 0 1")
    promo_moves = [m for m in board.legal_moves if m.promotion is not None]
    assert len(promo_moves) == 4
    promotion_pieces = {m.promotion for m in promo_moves}
    assert promotion_pieces == {chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT}
    for move in promo_moves:
        action = encode_move(move, board)
        decoded = decode_action(action, board)
        assert decoded == move


def test_underpromotion_capture_round_trip() -> None:
    board = chess.Board("1n6/P7/8/8/8/8/8/k1K5 w - - 0 1")
    promo_moves = [m for m in board.legal_moves if m.promotion is not None]
    assert any(m.promotion == chess.KNIGHT and m.to_square == chess.B8 for m in promo_moves)
    for move in promo_moves:
        action = encode_move(move, board)
        decoded = decode_action(action, board)
        assert decoded == move


def test_castling_round_trip() -> None:
    board = chess.Board("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1")
    castles = [m for m in board.legal_moves if board.is_castling(m)]
    assert len(castles) == 2
    for move in castles:
        action = encode_move(move, board)
        decoded = decode_action(action, board)
        assert decoded == move


def test_decode_out_of_range_returns_none() -> None:
    board = chess.Board()
    assert decode_action(-1, board) is None
    assert decode_action(NUM_ACTIONS, board) is None
