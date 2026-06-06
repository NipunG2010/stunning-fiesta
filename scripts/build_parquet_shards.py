"""Convert a Lichess .pgn.zst monthly dump into parquet training shards.

Each row is one ply: the position before the move, the encoded action played,
the engine eval before and after the move, and Lichess's NAG (0 = unannotated,
2 = mistake, 4 = blunder, 6 = inaccuracy). Reward computation is deferred to
the trainer so we can sweep reward shapes without re-extracting.

Usage:
    python scripts/build_parquet_shards.py \\
        --src data/raw/lichess_db_standard_rated_2024-01.pgn.zst \\
        --dst data/processed \\
        --min-elo 1500 --max-elo 1700 --min-plies 20 \\
        --rows-per-shard 500000
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterator

import chess
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from chess_rl.data import (
    filter_game,
    iter_games_from_zst,
    parse_eval,
    parse_nag,
)
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


def _game_id(game: chess.pgn.Game) -> str:
    site = game.headers.get("Site", "")
    m = _SITE_ID_RE.search(site)
    return m.group(1) if m else site


def iter_rows(
    src: Path,
    min_elo: int,
    max_elo: int,
    min_plies: int,
) -> Iterator[dict]:
    for game in iter_games_from_zst(src):
        if not filter_game(
            game, min_elo=min_elo, max_elo=max_elo, min_plies=min_plies, require_eval=True
        ):
            continue
        gid = _game_id(game)
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


def _empty_buffer() -> dict[str, list]:
    return {f.name: [] for f in SCHEMA}


def _flush(buf: dict[str, list], dst_dir: Path, shard_idx: int) -> Path:
    table = pa.Table.from_pydict(buf, schema=SCHEMA)
    path = dst_dir / f"shard-{shard_idx:05d}.parquet"
    pq.write_table(table, path, compression="zstd")
    return path


def build_shards(
    src: Path,
    dst_dir: Path,
    min_elo: int = 1500,
    max_elo: int = 1700,
    min_plies: int = 20,
    rows_per_shard: int = 500_000,
    max_rows: int | None = None,
) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    buf = _empty_buffer()
    shard_idx = 0
    total = 0
    with tqdm(unit="ply", desc="extracting") as pbar:
        for row in iter_rows(src, min_elo, max_elo, min_plies):
            for k, v in row.items():
                buf[k].append(v)
            total += 1
            pbar.update(1)
            if len(buf["game_id"]) >= rows_per_shard:
                path = _flush(buf, dst_dir, shard_idx)
                pbar.write(f"wrote {path} ({len(buf['game_id'])} rows)")
                shard_idx += 1
                buf = _empty_buffer()
            if max_rows is not None and total >= max_rows:
                break
    if buf["game_id"]:
        path = _flush(buf, dst_dir, shard_idx)
        print(f"wrote {path} ({len(buf['game_id'])} rows)")
        shard_idx += 1
    return total


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--src", type=Path, required=True, help=".pgn.zst input file")
    p.add_argument("--dst", type=Path, required=True, help="output directory for shards")
    p.add_argument("--min-elo", type=int, default=1500)
    p.add_argument("--max-elo", type=int, default=1700)
    p.add_argument("--min-plies", type=int, default=20)
    p.add_argument("--rows-per-shard", type=int, default=500_000)
    p.add_argument("--max-rows", type=int, default=None, help="stop early (for dry runs)")
    args = p.parse_args()
    if not args.src.is_file():
        sys.exit(f"Not a file: {args.src}")
    total = build_shards(
        src=args.src,
        dst_dir=args.dst,
        min_elo=args.min_elo,
        max_elo=args.max_elo,
        min_plies=args.min_plies,
        rows_per_shard=args.rows_per_shard,
        max_rows=args.max_rows,
    )
    print(f"\nTotal plies written: {total}")


if __name__ == "__main__":
    main()
