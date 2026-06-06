"""Stream a Lichess monthly .pgn.zst straight from the CDN into parquet shards.

Disk usage stays at the size of the active shard buffer (~50-200 MB) plus the
parquet output, never the full 30 GB compressed file. The download is
sequential because zstd does not support random access -- but you can stop
early with --max-kept-games once enough filtered games have been collected,
and only that fraction of the file is actually transferred.

Typical small extract (1500-1700 Elo, ~50k filtered games -> ~2M plies,
~1.5 GB parquet, takes ~30 minutes on a Kaggle notebook):

    python scripts/stream_lichess_to_shards.py \\
        --month 2024-01 --dst data/processed --max-kept-games 50000

Tiny smoke run (a couple of MB of compressed data; useful as a first try):

    python scripts/stream_lichess_to_shards.py \\
        --month 2024-01 --dst /tmp/probe --max-kept-games 50
"""

from __future__ import annotations

import argparse
import io
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterator

import chess.pgn
import zstandard as zstd
from tqdm import tqdm

from chess_rl.shards import iter_rows_from_games, write_shards

LICHESS_URL_TEMPLATE = (
    "https://database.lichess.org/standard/lichess_db_standard_rated_{month}.pgn.zst"
)


class _ProgressByteReader:
    """Wrap a binary stream so each read() updates a tqdm bar."""

    def __init__(self, inner, pbar: tqdm, cap_bytes: int | None = None) -> None:
        self._inner = inner
        self._pbar = pbar
        self._cap = cap_bytes
        self._consumed = 0

    def read(self, size: int = -1) -> bytes:
        if self._cap is not None:
            remaining = self._cap - self._consumed
            if remaining <= 0:
                return b""
            if size < 0 or size > remaining:
                size = remaining
        data = self._inner.read(size)
        self._consumed += len(data)
        self._pbar.update(len(data))
        return data


def iter_games_from_url(url: str, cap_bytes: int | None = None) -> Iterator[chess.pgn.Game]:
    req = urllib.request.Request(url, headers={"User-Agent": "stunning-fiesta/0.1"})
    response = urllib.request.urlopen(req)
    content_length = response.headers.get("Content-Length")
    total = int(content_length) if content_length else None
    if cap_bytes is not None and total is not None:
        total = min(total, cap_bytes)

    pbar = tqdm(total=total, unit="B", unit_scale=True, desc="downloading", position=0)
    try:
        wrapped = _ProgressByteReader(response, pbar, cap_bytes=cap_bytes)
        dctx = zstd.ZstdDecompressor()
        decompressed = dctx.stream_reader(wrapped)
        text = io.TextIOWrapper(decompressed, encoding="utf-8", errors="replace")
        while True:
            try:
                game = chess.pgn.read_game(text)
            except (zstd.ZstdError, EOFError, OSError):
                return
            if game is None:
                return
            yield game
    finally:
        pbar.close()
        response.close()


def stream_and_shard(
    month: str,
    dst_dir: Path,
    min_elo: int = 1500,
    max_elo: int = 1700,
    min_plies: int = 20,
    rows_per_shard: int = 500_000,
    max_kept_games: int | None = None,
    max_rows: int | None = None,
    cap_bytes: int | None = None,
    url: str | None = None,
) -> int:
    if url is None:
        url = LICHESS_URL_TEMPLATE.format(month=month)
    print(f"streaming {url}", file=sys.stderr)
    games = iter_games_from_url(url, cap_bytes=cap_bytes)
    rows = iter_rows_from_games(
        games,
        min_elo=min_elo,
        max_elo=max_elo,
        min_plies=min_plies,
        max_kept_games=max_kept_games,
    )
    with tqdm(unit="ply", desc="extracting", position=1) as pbar:
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
    p.add_argument("--month", required=True, help="YYYY-MM (e.g. 2024-01)")
    p.add_argument("--dst", type=Path, required=True, help="output directory for shards")
    p.add_argument("--min-elo", type=int, default=1500)
    p.add_argument("--max-elo", type=int, default=1700)
    p.add_argument("--min-plies", type=int, default=20)
    p.add_argument("--rows-per-shard", type=int, default=500_000)
    p.add_argument(
        "--max-kept-games",
        type=int,
        default=None,
        help="stop after this many filtered games (the main knob for capping disk + bandwidth)",
    )
    p.add_argument("--max-rows", type=int, default=None, help="hard cap on total plies written")
    p.add_argument(
        "--cap-mb",
        type=int,
        default=None,
        help="hard cap on bytes downloaded (MB); useful for smoke tests",
    )
    p.add_argument(
        "--url",
        default=None,
        help="override the Lichess URL (mostly for testing)",
    )
    args = p.parse_args()
    cap_bytes = args.cap_mb * 1024 * 1024 if args.cap_mb else None
    total = stream_and_shard(
        month=args.month,
        dst_dir=args.dst,
        min_elo=args.min_elo,
        max_elo=args.max_elo,
        min_plies=args.min_plies,
        rows_per_shard=args.rows_per_shard,
        max_kept_games=args.max_kept_games,
        max_rows=args.max_rows,
        cap_bytes=cap_bytes,
        url=args.url,
    )
    print(f"\nTotal plies written: {total}", file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except urllib.error.HTTPError as e:
        sys.exit(f"HTTP error: {e}")
    except urllib.error.URLError as e:
        sys.exit(f"Network error: {e}")
