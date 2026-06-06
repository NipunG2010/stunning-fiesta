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

Pull a month of Lichess data:

```bash
./scripts/download_lichess.sh 2024-01 data/raw
```

Open `notebooks/01_data_smoke.ipynb` to verify the encoder + PGN parsing pipeline (runs in seconds on CPU, no real data needed).

## Repo layout

```
src/chess_rl/
  encoding.py        board -> (18, 8, 8) tensor; move <-> action index in [0, 4672)
  data.py            streaming PGN reader, filter, eval/NAG extraction
  utils/stockfish.py FEN-keyed Stockfish eval cache
scripts/
  download_lichess.sh
  push_dataset_to_kaggle.py   CLI backend; MCP backend slot reserved
notebooks/
  01_data_smoke.ipynb
tests/
  test_encoding.py   round-trip + plane layout, 120+ positions
```

## Compute

Primary: Kaggle notebooks (30 GPU-h/week, T4 16GB). Colab free as overflow. Stockfish eval cache runs on CPU and is shared across runs via a Kaggle Dataset.
