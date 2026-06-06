"""Convert a local Lichess .pgn.zst into parquet training shards.

For the streaming-from-URL variant that never lands the .zst on disk, see
`stream_lichess_to_shards.py`.

Usage:
    python scripts/build_parquet_shards.py \\
        --src data/raw/lichess_db_standard_rated_2024-01.pgn.zst \\
        --dst data/processed \\
        --min-elo 1500 --max-elo 1700 --min-plies 20 \\
        --rows-per-shard 500000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

from chess_rl.data import iter_games_from_zst
from chess_rl.shards import iter_rows_from_games, write_shards


def build_shards(
    src: Path,
    dst_dir: Path,
    min_elo: int = 1500,
    max_elo: int = 1700,
    min_plies: int = 20,
    rows_per_shard: int = 500_000,
    max_rows: int | None = None,
    max_kept_games: int | None = None,
) -> int:
    games = iter_games_from_zst(src)
    rows = iter_rows_from_games(
        games,
        min_elo=min_elo,
        max_elo=max_elo,
        min_plies=min_plies,
        max_kept_games=max_kept_games,
    )
    with tqdm(unit="ply", desc="extracting") as pbar:
        return write_shards(
            rows,
            dst_dir=dst_dir,
            rows_per_shard=rows_per_shard,
            max_rows=max_rows,
            progress=pbar,
        )


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
    p.add_argument("--max-kept-games", type=int, default=None, help="stop after N filtered games")
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
        max_kept_games=args.max_kept_games,
    )
    print(f"\nTotal plies written: {total}")


if __name__ == "__main__":
    main()
