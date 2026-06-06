# stunning-fiesta

Offline RL on human chess blunders. Train a policy that plays like a 1500–1700 Lichess human but avoids the blunder-class moves humans at that rating make.

See [`proposal.md`](proposal.md) for the research framing and [`plan.md`](plan.md) for the 10-week build plan.

## Status

| Week | Milestone | Status |
|------|-----------|--------|
| 1 | Data pipeline, board encoder, Stockfish wrapper | done |
| 2 | BC baseline | next |
| 3–4 | IQL | |
| 5 | Replay-weighted variant (stretch) | |
| 6 | Evaluation harness + tournaments | |
| 7 | Pareto sweep | |
| 8 | Ablations | |
| 9–10 | Writing & submission | |

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Build parquet training shards from Lichess. **Recommended: stream straight into shards** so the 30 GB compressed dump never lands on disk:

```bash
python scripts/stream_lichess_to_shards.py \
    --month 2024-01 --dst data/processed \
    --max-kept-games 50000
```

Disk usage stays at the active shard buffer (~50–200 MB) plus the final parquet output. `--max-kept-games` caps how many filtered games to keep — and the script stops downloading after that, so a small extract only pulls a fraction of the file.

If you'd rather download the .zst once and process locally (e.g. you'll re-extract with different filters later):

```bash
./scripts/download_lichess.sh 2024-01 data/raw
python scripts/build_parquet_shards.py \
    --src data/raw/lichess_db_standard_rated_2024-01.pgn.zst \
    --dst data/processed
```

Open `notebooks/01_data_smoke.ipynb` to verify the encoder + PGN parsing pipeline (runs in seconds on CPU, no real data needed).

To run on Kaggle (recommended for the free GPU quota), see [`KAGGLE.md`](KAGGLE.md).

## Repo layout

```
src/chess_rl/
  encoding.py        board -> (18, 8, 8) tensor; move <-> action index in [0, 4672)
  data.py            streaming PGN reader, filter, eval/NAG extraction
  utils/stockfish.py FEN-keyed Stockfish eval cache
scripts/
  download_lichess.sh             pull one month from database.lichess.org
  build_parquet_shards.py         local .pgn.zst -> parquet shards
  stream_lichess_to_shards.py     stream from URL -> shards (no 30 GB on disk)
  push_dataset_to_kaggle.py       CLI backend; MCP backend slot reserved
notebooks/
  01_data_smoke.ipynb
tests/
  test_encoding.py            round-trip + plane layout, 120+ positions
  test_build_shards.py        local shard builder, synthetic PGN
  test_stream_shards.py       streaming shard builder, local HTTP + zstd
```

## Compute

Primary: Kaggle notebooks (30 GPU-h/week, T4 16GB). Colab free as overflow. Stockfish eval cache runs on CPU and is shared across runs via a Kaggle Dataset.
