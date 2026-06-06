"""End-to-end smoke test for the parquet shard builder on a synthetic PGN."""

from __future__ import annotations

import sys
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import zstandard as zstd

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from build_parquet_shards import build_shards  # noqa: E402

PGN = """[Event "Rated Blitz game"]
[Site "https://lichess.org/abcd1234"]
[White "alice"]
[Black "bob"]
[Result "1-0"]
[WhiteElo "1600"]
[BlackElo "1620"]
[TimeControl "300+0"]
[Termination "Normal"]

1. e4 { [%eval 0.2] } 1... e5 { [%eval 0.3] } 2. Nf3 { [%eval 0.25] } 2... Nc6 { [%eval 0.3] } 3. Bb5 { [%eval 0.32] } 3... a6 { [%eval 0.35] } 4. Ba4 { [%eval 0.3] } 4... Nf6 { [%eval 0.35] } 5. O-O { [%eval 0.4] } 5... Be7 { [%eval 0.4] } 6. Re1 { [%eval 0.4] } 6... b5 { [%eval 0.45] } 7. Bb3 { [%eval 0.4] } 7... d6 { [%eval 0.5] } 8. c3 { [%eval 0.4] } 8... O-O { [%eval 0.45] } 9. h3 { [%eval 0.4] } 9... Nb8?? { [%eval 5.2] } 10. d4 1-0
"""


def _write_zst(tmp_path: Path) -> Path:
    src = tmp_path / "tiny.pgn.zst"
    src.write_bytes(zstd.ZstdCompressor().compress(PGN.encode()))
    return src


def test_build_shards_produces_parquet(tmp_path: Path) -> None:
    src = _write_zst(tmp_path)
    dst = tmp_path / "shards"
    total = build_shards(src=src, dst_dir=dst, min_plies=10, rows_per_shard=1000)
    assert total > 0

    shards = sorted(dst.glob("*.parquet"))
    assert len(shards) == 1

    table = pq.read_table(shards[0])
    assert table.num_rows == total

    df = table.to_pandas()
    assert set(df["game_id"]) == {"abcd1234"}
    assert df["ply"].iloc[0] == 0
    assert df["ply"].is_monotonic_increasing
    assert df["white_to_move"].iloc[0] is True or df["white_to_move"].iloc[0] == True  # noqa: E712
    assert df["white_to_move"].iloc[1] in (False,)

    blunders = df[df["nag"] == 4]
    assert len(blunders) == 1
    assert blunders["eval_after_cp"].iloc[0] == pytest.approx(520.0)


def test_max_rows_early_stop(tmp_path: Path) -> None:
    src = _write_zst(tmp_path)
    dst = tmp_path / "shards"
    total = build_shards(src=src, dst_dir=dst, min_plies=10, rows_per_shard=1000, max_rows=5)
    assert total == 5
