"""Board and move encoding for the chess policy.

The board is encoded as a (18, 8, 8) float tensor: 12 piece planes + 4 castling
flags + 1 side-to-move + 1 en-passant target. Moves are encoded as integers in
[0, 4672) using the AlphaZero-style 73-plane layout, indexed by source square.
"""

from __future__ import annotations

import chess
import numpy as np

NUM_SQUARES = 64
NUM_MOVE_PLANES = 73
NUM_ACTIONS = NUM_SQUARES * NUM_MOVE_PLANES
INPUT_PLANES = 18

PIECE_TO_PLANE: dict[tuple[int, bool], int] = {
    (chess.PAWN, chess.WHITE): 0,
    (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK, chess.WHITE): 3,
    (chess.QUEEN, chess.WHITE): 4,
    (chess.KING, chess.WHITE): 5,
    (chess.PAWN, chess.BLACK): 6,
    (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK, chess.BLACK): 9,
    (chess.QUEEN, chess.BLACK): 10,
    (chess.KING, chess.BLACK): 11,
}

# (delta_rank, delta_file) for each of 8 sliding directions.
QUEEN_DIRS: list[tuple[int, int]] = [
    (1, 0), (1, 1), (0, 1), (-1, 1),
    (-1, 0), (-1, -1), (0, -1), (1, -1),
]

KNIGHT_DELTAS: list[tuple[int, int]] = [
    (2, 1), (1, 2), (-1, 2), (-2, 1),
    (-2, -1), (-1, -2), (1, -2), (2, -1),
]

UNDERPROMO_PIECES: list[int] = [chess.KNIGHT, chess.BISHOP, chess.ROOK]
UNDERPROMO_DELTAS_WHITE: list[tuple[int, int]] = [(1, -1), (1, 0), (1, 1)]
UNDERPROMO_DELTAS_BLACK: list[tuple[int, int]] = [(-1, -1), (-1, 0), (-1, 1)]


def encode_board(board: chess.Board) -> np.ndarray:
    planes = np.zeros((INPUT_PLANES, 8, 8), dtype=np.float32)
    for square, piece in board.piece_map().items():
        plane = PIECE_TO_PLANE[(piece.piece_type, piece.color)]
        rank, file = divmod(square, 8)
        planes[plane, rank, file] = 1.0
    if board.has_kingside_castling_rights(chess.WHITE):
        planes[12, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        planes[13, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        planes[14, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        planes[15, :, :] = 1.0
    if board.turn == chess.WHITE:
        planes[16, :, :] = 1.0
    if board.ep_square is not None:
        rank, file = divmod(board.ep_square, 8)
        planes[17, rank, file] = 1.0
    return planes


def _queen_plane(dr: int, df: int) -> int | None:
    for i, (qdr, qdf) in enumerate(QUEEN_DIRS):
        for d in range(1, 8):
            if qdr * d == dr and qdf * d == df:
                return i * 7 + (d - 1)
    return None


def encode_move(move: chess.Move, board: chess.Board) -> int:
    from_sq = move.from_square
    to_sq = move.to_square
    from_rank, from_file = divmod(from_sq, 8)
    to_rank, to_file = divmod(to_sq, 8)
    dr = to_rank - from_rank
    df = to_file - from_file

    plane: int | None = None

    if move.promotion is not None and move.promotion != chess.QUEEN:
        piece = board.piece_at(from_sq)
        if piece is None:
            raise ValueError(f"No piece at {chess.square_name(from_sq)} for move {move.uci()}")
        deltas = UNDERPROMO_DELTAS_WHITE if piece.color == chess.WHITE else UNDERPROMO_DELTAS_BLACK
        if (dr, df) not in deltas:
            raise ValueError(f"Illegal underpromotion delta for {move.uci()}")
        delta_idx = deltas.index((dr, df))
        piece_idx = UNDERPROMO_PIECES.index(move.promotion)
        plane = 64 + piece_idx * 3 + delta_idx
    elif (dr, df) in KNIGHT_DELTAS:
        plane = 56 + KNIGHT_DELTAS.index((dr, df))
    else:
        plane = _queen_plane(dr, df)

    if plane is None:
        raise ValueError(f"Cannot encode move {move.uci()} from {board.fen()}")
    return from_sq * NUM_MOVE_PLANES + plane


def decode_action(action: int, board: chess.Board) -> chess.Move | None:
    if not (0 <= action < NUM_ACTIONS):
        return None
    from_sq, plane = divmod(action, NUM_MOVE_PLANES)
    from_rank, from_file = divmod(from_sq, 8)

    if plane < 56:
        dir_idx, dist_idx = divmod(plane, 7)
        dr, df = QUEEN_DIRS[dir_idx]
        distance = dist_idx + 1
        to_rank = from_rank + dr * distance
        to_file = from_file + df * distance
        if not (0 <= to_rank < 8 and 0 <= to_file < 8):
            return None
        to_sq = to_rank * 8 + to_file
        promotion: int | None = None
        piece = board.piece_at(from_sq)
        # A pawn reaching the last rank via a queen-plane move means queen promotion;
        # underpromotions go through the 64-72 plane range.
        if piece is not None and piece.piece_type == chess.PAWN:
            if (piece.color == chess.WHITE and to_rank == 7) or (
                piece.color == chess.BLACK and to_rank == 0
            ):
                promotion = chess.QUEEN
        return chess.Move(from_sq, to_sq, promotion=promotion)

    if plane < 64:
        dr, df = KNIGHT_DELTAS[plane - 56]
        to_rank = from_rank + dr
        to_file = from_file + df
        if not (0 <= to_rank < 8 and 0 <= to_file < 8):
            return None
        return chess.Move(from_sq, to_rank * 8 + to_file)

    underpromo_idx = plane - 64
    piece_idx, delta_idx = divmod(underpromo_idx, 3)
    promotion = UNDERPROMO_PIECES[piece_idx]
    piece = board.piece_at(from_sq)
    if piece is None:
        return None
    deltas = UNDERPROMO_DELTAS_WHITE if piece.color == chess.WHITE else UNDERPROMO_DELTAS_BLACK
    dr, df = deltas[delta_idx]
    to_rank = from_rank + dr
    to_file = from_file + df
    if not (0 <= to_rank < 8 and 0 <= to_file < 8):
        return None
    return chess.Move(from_sq, to_rank * 8 + to_file, promotion=promotion)


def legal_action_mask(board: chess.Board) -> np.ndarray:
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    for move in board.legal_moves:
        mask[encode_move(move, board)] = True
    return mask
