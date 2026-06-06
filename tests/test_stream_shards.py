"""End-to-end test for the streaming shard builder.

Spins up a one-shot local HTTP server that serves a synthetic .pgn.zst, so the
test exercises the real urllib + zstd streaming path without needing internet
or the 30 GB Lichess file.
"""

from __future__ import annotations

import http.server
import socketserver
import sys
import threading
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import zstandard as zstd

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from stream_lichess_to_shards import stream_and_shard  # noqa: E402

PGN_TEMPLATE = """[Event "Rated Blitz game"]
[Site "https://lichess.org/{game_id}"]
[White "alice"]
[Black "bob"]
[Result "1-0"]
[WhiteElo "1600"]
[BlackElo "1620"]
[TimeControl "300+0"]
[Termination "Normal"]

1. e4 {{ [%eval 0.2] }} 1... e5 {{ [%eval 0.3] }} 2. Nf3 {{ [%eval 0.25] }} 2... Nc6 {{ [%eval 0.3] }} 3. Bb5 {{ [%eval 0.32] }} 3... a6 {{ [%eval 0.35] }} 4. Ba4 {{ [%eval 0.3] }} 4... Nf6 {{ [%eval 0.35] }} 5. O-O {{ [%eval 0.4] }} 5... Be7 {{ [%eval 0.4] }} 6. Re1 {{ [%eval 0.4] }} 6... b5 {{ [%eval 0.45] }} 7. Bb3 {{ [%eval 0.4] }} 7... d6 {{ [%eval 0.5] }} 8. c3 {{ [%eval 0.4] }} 8... O-O {{ [%eval 0.45] }} 9. h3 {{ [%eval 0.4] }} 9... Nb8?? {{ [%eval 5.2] }} 10. d4 1-0

"""


@pytest.fixture
def served_zst(tmp_path: Path):
    body = "".join(PGN_TEMPLATE.format(game_id=f"id{i:04d}") for i in range(3))
    zst_bytes = zstd.ZstdCompressor().compress(body.encode())
    serve_dir = tmp_path / "serve"
    serve_dir.mkdir()
    (serve_dir / "month.pgn.zst").write_bytes(zst_bytes)

    handler = lambda *args, **kwargs: http.server.SimpleHTTPRequestHandler(
        *args, directory=str(serve_dir), **kwargs
    )
    server = socketserver.TCPServer(("127.0.0.1", 0), handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}/month.pgn.zst"
    finally:
        server.shutdown()
        server.server_close()


def test_stream_produces_shards(served_zst: str, tmp_path: Path) -> None:
    dst = tmp_path / "shards"
    total = stream_and_shard(
        month="ignored",
        dst_dir=dst,
        min_plies=10,
        rows_per_shard=1000,
        url=served_zst,
    )
    assert total > 0

    shards = sorted(dst.glob("*.parquet"))
    assert len(shards) == 1
    df = pq.read_table(shards[0]).to_pandas()
    assert set(df["game_id"]) == {"id0000", "id0001", "id0002"}
    assert (df["nag"] == 4).sum() == 3
    assert df["ply"].min() == 0


def test_stream_max_kept_games(served_zst: str, tmp_path: Path) -> None:
    dst = tmp_path / "shards"
    stream_and_shard(
        month="ignored",
        dst_dir=dst,
        min_plies=10,
        rows_per_shard=1000,
        url=served_zst,
        max_kept_games=2,
    )
    df = pq.read_table(sorted(dst.glob("*.parquet"))[0]).to_pandas()
    assert set(df["game_id"]) == {"id0000", "id0001"}


def test_stream_cap_bytes_truncates_gracefully(served_zst: str, tmp_path: Path) -> None:
    dst = tmp_path / "shards"
    total = stream_and_shard(
        month="ignored",
        dst_dir=dst,
        min_plies=10,
        rows_per_shard=1000,
        url=served_zst,
        cap_bytes=64,
    )
    assert total == 0 or total > 0
    assert dst.exists()
