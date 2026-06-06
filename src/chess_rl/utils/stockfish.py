"""Stockfish subprocess wrapper with a FEN-keyed eval cache.

Evaluations are expensive (~0.5s at depth 18). The cache is a shelve database
so it survives across notebook restarts on Kaggle.
"""

from __future__ import annotations

import shelve
from pathlib import Path
from typing import Self

import chess
import chess.engine

MATE_CP = 10000


class StockfishEvaluator:
    def __init__(
        self,
        binary: str = "stockfish",
        cache_path: Path | str | None = None,
        depth: int = 18,
        threads: int = 1,
    ) -> None:
        self.depth = depth
        self.engine = chess.engine.SimpleEngine.popen_uci(binary)
        self.engine.configure({"Threads": threads})
        self._cache = shelve.open(str(cache_path)) if cache_path else None

    def evaluate(self, board: chess.Board) -> float:
        key = board.fen()
        if self._cache is not None and key in self._cache:
            return self._cache[key]
        info = self.engine.analyse(board, chess.engine.Limit(depth=self.depth))
        score = info["score"].white()
        if score.is_mate():
            cp = float(MATE_CP if score.mate() > 0 else -MATE_CP)
        else:
            cp = float(score.score(mate_score=MATE_CP))
        if self._cache is not None:
            self._cache[key] = cp
        return cp

    def close(self) -> None:
        try:
            self.engine.quit()
        finally:
            if self._cache is not None:
                self._cache.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
