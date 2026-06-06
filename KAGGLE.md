# Running on Kaggle

This repo is designed to run inside Kaggle notebooks for the GPU quota (30 T4-hours/week free). This guide walks through the click-by-click setup.

## One-time setup

1. **Create a Kaggle account** at https://www.kaggle.com and verify your phone number — required to enable GPU and Internet on notebooks.
2. **Generate an API token** (only needed if you want to run `scripts/push_dataset_to_kaggle.py` from anywhere): Kaggle → Account → API → *Create New Token* → downloads `kaggle.json`. Place at `~/.kaggle/kaggle.json` (chmod 600).

## Running `notebooks/01_data_smoke.ipynb`

1. Go to https://www.kaggle.com/code and click **New Notebook**.
2. In the right sidebar:
   - **Settings → Internet → On** (required so the first cell can `git clone`).
   - **Accelerator → None** (this notebook is CPU-only).
3. **File → Import Notebook → GitHub** and paste:
   ```
   https://github.com/NipunG2010/stunning-fiesta/blob/master/notebooks/01_data_smoke.ipynb
   ```
   Or download `01_data_smoke.ipynb` and upload it.
4. Click **Run All**. Total runtime is under a minute.

The first cell auto-detects the Kaggle environment, clones the repo into `/kaggle/working/stunning-fiesta`, installs the package, and adds it to `sys.path`. Subsequent cells exercise the encoder and the PGN parser on a synthetic Lichess-format game.

## Running on the real Lichess data

The full month is ~30 GB compressed, which exceeds a Kaggle notebook's `/kaggle/working` budget. Two ways to make it fit:

### Option A — Attach an existing Kaggle Lichess Dataset

Several users mirror the Lichess monthly dumps as Kaggle Datasets. Search "lichess pgn 2024" under Datasets. Add the dataset to your notebook via **Add Input → Datasets**, then point `iter_games_from_zst` at the path under `/kaggle/input/<dataset-slug>/`.

### Option B — Stream-process and shard locally, then upload as a Dataset

1. Run `scripts/download_lichess.sh 2024-01 data/raw` on a machine with disk space.
2. Run `scripts/build_parquet_shards.py` (see Week 2) to produce ~200 MB shards.
3. Publish the shards as a Kaggle Dataset with `scripts/push_dataset_to_kaggle.py --backend cli`.
4. Attach that dataset to subsequent training notebooks.

The processed shards are ~1 GB total — well within Kaggle's per-dataset and per-notebook size limits.

## Troubleshooting

- **`git: command not found`** — Kaggle has git pre-installed; the more likely cause is Internet being off in Settings.
- **`pip install` is slow** — normal on the first run (Torch, pyarrow). Subsequent runs in the same notebook session reuse the install.
- **`ModuleNotFoundError: chess_rl`** — the first cell didn't run, or you skipped the `sys.path.insert`. Re-run cell 1.
- **Kernel runs out of memory** — the smoke notebook uses < 200 MB. If you hit OOM here, the kernel is likely sharing the box with another notebook of yours; close the others.
