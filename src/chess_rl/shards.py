"""Build training parquet shards from a games iterator.

Decoupled from the source so the same row-extraction and shard-writing logic
can run against either a local .pgn.zst file (`scripts/build_parquet_shards.py`)
or an HTTP-streamed file (`scripts/stream_lichess_to_shards.py`) without
landing the full 30 GB on disk.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Iterator

import chess
import chess.pgn
import pyarrow as pa
import pyarrow.parquet as pq

from chess_rl.data import filter_game, parse_eval, parse_nag
from chess_rl.encoding import encode_move

SCHEMA = pa.schema(
    [
        pa.field("game_id", pa.string()),
        pa.field("ply", pa.int16()),
        pa.field("fen", pa.string()),
        pa.field("action_id", pa.int32()),
        pa.field("eval_before_cp", pa.float32()),
        pa.field("eval_after_cp", pa.float32()),
        pa.field("nag", pa.int8()),
        pa.field("white_to_move", pa.bool_()),
    ]
)

_SITE_ID_RE = re.compile(r"lichess\.org/([A-Za-z0-9]+)")


def game_id(game: chess.pgn.Game) -> str:
    site = game.headers.get("Site", "")
    m = _SITE_ID_RE.search(site)
    return m.group(1) if m else site


def iter_rows_from_games(
    games: Iterable[chess.pgn.Game],
    min_elo: int = 1500,
    max_elo: int = 1700,
    min_plies: int = 20,
    max_kept_games: int | None = None,
) -> Iterator[dict]:
    kept = 0
    for game in games:
        if not filter_game(
            game, min_elo=min_elo, max_elo=max_elo, min_plies=min_plies, require_eval=True
        ):
            continue
        gid = game_id(game)
        board = game.board()
        prev_eval: float | None = None
        node = game
        ply = 0
        while node.variations:
            node = node.variations[0]
            move = node.move
            cur_eval = parse_eval(node.comment or "")
            nag = parse_nag(node.nags) or 0
            try:
                action_id = encode_move(move, board)
            except ValueError:
                board.push(move)
                ply += 1
                prev_eval = cur_eval
                continue
            yield {
                "game_id": gid,
                "ply": ply,
                "fen": board.fen(),
                "action_id": action_id,
                "eval_before_cp": float("nan") if prev_eval is None else prev_eval,
                "eval_after_cp": float("nan") if cur_eval is None else cur_eval,
                "nag": nag,
                "white_to_move": board.turn == chess.WHITE,
            }
            board.push(move)
            prev_eval = cur_eval
            ply += 1
        kept += 1
        if max_kept_games is not None and kept >= max_kept_games:
            return


def _empty_buffer() -> dict[str, list]:
    return {f.name: [] for f in SCHEMA}


def _flush(buf: dict[str, list], dst_dir: Path, shard_idx: int) -> Path:
    table = pa.Table.from_pydict(buf, schema=SCHEMA)
    path = dst_dir / f"shard-{shard_idx:05d}.parquet"
    pq.write_table(table, path, compression="zstd")
    return path


def write_shards(
    rows: Iterable[dict],
    dst_dir: Path,
    rows_per_shard: int = 500_000,
    max_rows: int | None = None,
    progress=None,
) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    buf = _empty_buffer()
    shard_idx = 0
    total = 0
    for row in rows:
        for k, v in row.items():
            buf[k].append(v)
        total += 1
        if progress is not None:
            progress.update(1)
        if len(buf["game_id"]) >= rows_per_shard:
            path = _flush(buf, dst_dir, shard_idx)
            if progress is not None:
                progress.write(f"wrote {path} ({len(buf['game_id'])} rows)")
            else:
                print(f"wrote {path} ({len(buf['game_id'])} rows)")
            shard_idx += 1
            buf = _empty_buffer()
        if max_rows is not None and total >= max_rows:
            break
    if buf["game_id"]:
        path = _flush(buf, dst_dir, shard_idx)
        print(f"wrote {path} ({len(buf['game_id'])} rows)")
        shard_idx += 1
    return total
