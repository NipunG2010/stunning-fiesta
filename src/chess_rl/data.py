"""Streaming PGN reader for the Lichess open database.

The Lichess monthly dumps are zstandard-compressed PGN containing per-move
engine evals (`[%eval ...]`) and Lichess's own blunder NAGs on a subset of
games. We stream-decompress and yield filtered games without ever holding
the full uncompressed file in memory.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import chess
import chess.pgn
import zstandard as zstd

NAG_INACCURACY = 6
NAG_MISTAKE = 2
NAG_BLUNDER = 4

_EVAL_PAWN_RE = re.compile(r"\[%eval ([-+]?[\d.]+)\]")
_EVAL_MATE_RE = re.compile(r"\[%eval #([-+]?\d+)\]")


@dataclass
class MoveRecord:
    move: chess.Move
    eval_cp: float | None
    nag: int | None


def iter_games_from_zst(path: Path | str) -> Iterator[chess.pgn.Game]:
    path = Path(path)
    with open(path, "rb") as f:
        dctx = zstd.ZstdDecompressor()
        stream = dctx.stream_reader(f)
        text = io.TextIOWrapper(stream, encoding="utf-8")
        while True:
            game = chess.pgn.read_game(text)
            if game is None:
                break
            yield game


def filter_game(
    game: chess.pgn.Game,
    min_elo: int = 1500,
    max_elo: int = 1700,
    min_plies: int = 20,
    require_eval: bool = True,
) -> bool:
    headers = game.headers
    try:
        white_elo = int(headers.get("WhiteElo", 0))
        black_elo = int(headers.get("BlackElo", 0))
    except ValueError:
        return False
    if not (min_elo <= white_elo <= max_elo and min_elo <= black_elo <= max_elo):
        return False
    if headers.get("Termination", "").lower() == "abandoned":
        return False

    node = game
    plies = 0
    saw_eval = False
    while node.variations:
        node = node.variations[0]
        plies += 1
        if not saw_eval and node.comment and "%eval" in node.comment:
            saw_eval = True
    if plies < min_plies:
        return False
    if require_eval and not saw_eval:
        return False
    return True


def parse_eval(comment: str) -> float | None:
    if not comment:
        return None
    m = _EVAL_PAWN_RE.search(comment)
    if m:
        return float(m.group(1)) * 100.0
    m = _EVAL_MATE_RE.search(comment)
    if m:
        return 10000.0 if int(m.group(1)) > 0 else -10000.0
    return None


def parse_nag(nags) -> int | None:
    for n in nags or ():
        if n in (NAG_BLUNDER, NAG_MISTAKE, NAG_INACCURACY):
            return n
    return None


def iter_moves(game: chess.pgn.Game) -> Iterator[MoveRecord]:
    node = game
    while node.variations:
        node = node.variations[0]
        yield MoveRecord(
            move=node.move,
            eval_cp=parse_eval(node.comment or ""),
            nag=parse_nag(node.nags),
        )
