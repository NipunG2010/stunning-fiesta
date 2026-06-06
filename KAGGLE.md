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

### Option A — Stream directly into parquet shards (recommended)

This avoids ever holding the 30 GB compressed file on disk. From a Kaggle notebook with Internet enabled:

```python
!python /kaggle/working/stunning-fiesta/scripts/stream_lichess_to_shards.py \
    --month 2024-01 \
    --dst /kaggle/working/shards \
    --max-kept-games 50000
```

The script decompresses on the fly and writes parquet shards incrementally. `--max-kept-games 50000` produces about 2M training plies (~1.5 GB of parquet) and stops the download once enough filtered games have been collected — so only a fraction of the 30 GB is actually transferred. For a first-try smoke check, use `--max-kept-games 50 --cap-mb 50`.

After it finishes, publish the shards as a Kaggle Dataset (`scripts/push_dataset_to_kaggle.py`) so subsequent training notebooks can attach them via **Add Input → Datasets** without re-running the extract.

### Option B — Attach an existing Kaggle Lichess Dataset

Several users mirror the Lichess monthly dumps as Kaggle Datasets. Search "lichess pgn 2024" under Datasets, then run `scripts/build_parquet_shards.py --src /kaggle/input/<slug>/<file>.pgn.zst --dst /kaggle/working/shards` to get the same parquet output.

## Troubleshooting

- **`git: command not found`** — Kaggle has git pre-installed; the more likely cause is Internet being off in Settings.
- **`pip install` is slow** — normal on the first run (Torch, pyarrow). Subsequent runs in the same notebook session reuse the install.
- **`ModuleNotFoundError: chess_rl`** — the first cell didn't run, or you skipped the `sys.path.insert`. Re-run cell 1.
- **Kernel runs out of memory** — the smoke notebook uses < 200 MB. If you hit OOM here, the kernel is likely sharing the box with another notebook of yours; close the others.
