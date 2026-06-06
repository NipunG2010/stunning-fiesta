# Execution Plan: Offline RL on Human Chess Blunders

**Status:** draft v0.1 · **Companion to:** `proposal.md` · **Target submission:** ~10 weeks out

This plan turns the proposal into a concrete, week-by-week build. It is scoped to the constraints we actually have: free Kaggle / Colab T4, open-source data, no API budget, solo developer. Anything not on this plan is explicitly out of scope for v1.

---

## 0. Scope cuts vs the proposal

The proposal lists CQL + IQL × 3 Elo buckets × ablations. For v1 we cut to:

- **One Elo bucket:** 1500–1700 (largest population on Lichess, plenty of blunders).
- **One algorithm to ship:** IQL (more stable than CQL on discrete actions; cheaper hyperparam search).
- **One reward variant:** clipped Δeval + `??` penalty. Replay-weighting is a stretch goal.
- **One baseline pair:** BC on all moves + public Maia-1700 checkpoint. We add CQL only if IQL works.

We can re-expand once the v1 pipeline is green. Cutting scope is the single biggest risk reducer.

---

## 1. Repo layout (target)

```
stunning-fiesta/
├── proposal.md
├── plan.md
├── README.md
├── pyproject.toml          # uv-managed deps, pinned
├── data/
│   ├── raw/                # gitignored; Lichess .pgn.zst archives
│   ├── processed/          # gitignored; parquet shards of (state, action, eval, blunder_tag)
│   └── splits/             # versioned train/val/test position IDs (small, committed)
├── src/chess_rl/
│   ├── encoding.py         # board → 8x8x18 tensor; move → 4672-action index
│   ├── data.py             # PGN streaming, blunder labeling, parquet writer
│   ├── dataset.py          # torch IterableDataset over parquet shards
│   ├── models/
│   │   ├── resnet.py       # 6-block residual policy/value net
│   │   └── heads.py        # policy + value heads
│   ├── train/
│   │   ├── bc.py           # behavioral cloning trainer
│   │   └── iql.py          # IQL trainer (d3rlpy or hand-rolled)
│   ├── eval/
│   │   ├── blunder_rate.py # Stockfish-classified blunders on holdout
│   │   ├── agreement.py    # top-1 move-agreement vs human moves
│   │   └── tournament.py   # round-robin vs Maia + nerfed Stockfish
│   └── utils/
│       ├── stockfish.py    # subprocess wrapper, eval cache
│       └── maia.py         # load + query public Maia checkpoints
├── notebooks/
│   ├── 01_data_smoke.ipynb        # runs on Kaggle
│   ├── 02_bc_baseline.ipynb       # runs on Kaggle
│   ├── 03_iql_train.ipynb         # runs on Kaggle
│   └── 04_eval_tournament.ipynb   # runs on Kaggle
├── scripts/
│   ├── download_lichess.sh        # pulls one month from database.lichess.org
│   └── push_dataset_to_kaggle.py  # via Kaggle MCP
└── paper/
    └── main.tex            # workshop template (NeurIPS/ICLR Tiny)
```

---

## 2. Tech stack (pinned)

- **Language:** Python 3.11
- **Deps (uv-managed):** `torch`, `python-chess`, `pyarrow`, `zstandard`, `d3rlpy`, `wandb` (free tier), `numpy`, `pandas`, `tqdm`, `omegaconf`
- **External binaries:** Stockfish 16 (apt or static binary)
- **Compute:** Kaggle notebooks (30 GPU-h/week T4) as primary; Colab free as overflow
- **Experiment tracking:** Weights & Biases free tier (single project, public)
- **Versioning:** Git for code; Kaggle Datasets for processed data shards; Hugging Face model hub for released checkpoints

---

## 3. Kaggle workflow

Kaggle is our compute *and* our data versioning store. The MCP (once loaded) lets us automate the loop:

1. **Code** lives in this Git repo.
2. **Raw PGN → processed parquet shards** is run once locally or in a CPU-only Kaggle notebook, then published as a **Kaggle Dataset** (`nipung2010/lichess-blunder-shards-v1`).
3. **Training notebooks** declare that dataset as input, save model weights + W&B run IDs as **Kaggle Notebook outputs**, which we then publish as a second Kaggle Dataset (`nipung2010/chess-rl-checkpoints`).
4. **Evaluation notebook** consumes both the processed dataset and the checkpoints dataset, runs the tournament, and writes a results table.

Kaggle MCP tool calls we expect to make (will confirm names once the server is loaded):

- `kaggle.datasets.create` / `kaggle.datasets.version` — push processed shards and checkpoints.
- `kaggle.kernels.push` — upload + run a notebook from this repo on Kaggle GPU.
- `kaggle.kernels.status` / `kaggle.kernels.output` — poll runs, pull artifacts.

Until MCP is wired, the same flow runs via the `kaggle` CLI in `scripts/`. Both paths target identical artifact names so we can swap freely.

---

## 4. Data pipeline (Week 1)

**Source.** `https://database.lichess.org/standard/lichess_db_standard_rated_2024-01.pgn.zst` (~30 GB compressed, ~100 GB uncompressed). One month is sufficient for v1.

**Filter.** Keep games where both players are rated 1500–1700, blitz or rapid time control, finished without abort, at least 20 plies.

**Annotation.** Lichess PGNs already include `[%eval ...]` and `?`/`??` NAGs on a subset of games — those are the games we keep. We don't need to run Stockfish during data prep; we only run it during evaluation.

**Encoding.**
- State: `8×8×18` tensor (12 piece planes + 4 castling + 1 side-to-move + 1 en-passant column).
- Action: AlphaZero-style 73-plane move encoding → 4672 discrete actions, masked by legal moves at inference.
- Reward: `r_t = clip(-Δeval_centipawns / 100, -3, +1)` with extra `-1.0` if move tagged `??`.

**Output.** ~5M (state, action, reward, next_state, done, eval_before, eval_after, blunder_flag) tuples, written as zstd-compressed parquet shards (~200 MB each), uploaded as a Kaggle Dataset.

**Deliverable:** notebook `01_data_smoke.ipynb` that reads one shard and prints summary stats; CI-style sanity check that blunder rate in the data matches Lichess's published figure (~6% in this band).

---

## 5. Modeling (Weeks 2–5)

### Week 2 — BC baseline
- ResNet (6 blocks × 128 channels, ~1.5M params).
- Cross-entropy on (state → action) over all moves in the dataset.
- Target: ~45% top-1 agreement with held-out human moves in the same Elo band (Maia hits ~52% — we are not trying to beat it here, just sanity-check the pipeline).
- **Done when:** held-out top-1 agreement converges and W&B run is logged.

### Weeks 3–4 — IQL
- Implement IQL on the same dataset and the same network backbone:
  - Q-network: ResNet trunk + value head (per-action Q over 4672 actions, masked by legality).
  - V-network: separate value head trained with expectile regression at τ=0.7.
  - Policy: advantage-weighted regression on the BC policy, β=3.0 initial.
- Hyperparam sweep (small): β ∈ {1, 3, 10}, τ ∈ {0.7, 0.9}. 6 runs × 8 h each = 48 GPU-h ≈ 1.5 Kaggle weeks.
- **Done when:** at least one IQL run dominates BC on the Pareto frontier of (agreement, blunder rate) on a 50k-position validation slice.

### Week 5 — Replay weighting (stretch)
- Upsample states immediately preceding a `??` move (sampling weight ×5).
- Re-run the best IQL config. Cut if Week 4 ran long.

---

## 6. Evaluation harness (Week 6)

Three metrics, all on a frozen 50k-position holdout from a different Lichess month (2024-02):

1. **Blunder rate.** For each model move, run Stockfish (depth 18, ~0.5 s/position) on the resulting position and tag `??` per Lichess's threshold (Δeval > 3 pawns, not a forced sequence). Cache evals keyed by FEN. ~7 h CPU once, then free for future runs.
2. **Move-agreement.** Top-1 match against the move played by a 1500–1700 human in held-out games, bucketed by game phase.
3. **Playing strength.** 200-game tournaments (we don't need 1000 for a workshop): model vs Maia-1500, model vs Maia-1700, model vs Stockfish skill-5 depth-5. Report Elo difference with Bayeselo.

**Deliverable:** a single `results.csv` per model checkpoint, plus the Pareto plot (`agreement` vs `1 - blunder_rate`).

---

## 7. Risk register

| Risk | Likelihood | Cost if it hits | Mitigation |
|---|---|---|---|
| Lichess month doesn't yield enough rated-band blunders | Med | Re-filter; pull a second month | Pre-flight in Week 1 |
| Kaggle GPU quota exhausted mid-sweep | Med | Slip 1 week | Cut sweep to 3 configs; failover to Colab |
| IQL underperforms BC on both metrics | Low–Med | Pivot story | Frame as "RL is hard here, here's why" — still publishable |
| Stockfish eval cache too slow | Med | Slip eval week | Run on CPU in parallel with training, not after |
| Maia checkpoint format breaks `python-chess` integration | Low | 1 day debug | Smoke test in Week 1, not Week 6 |
| W&B free tier limits | Low | Annoying | Local TensorBoard fallback |

The single biggest derisk is the **Week-1 smoke test**: data pipeline + Maia loading + Stockfish eval cache all touched before any training. If any of those slip, we discover it cheap.

---

## 8. Definition of done (paper-shippable)

A workshop paper is shippable when, all on the same held-out month:

1. We have an IQL checkpoint that, at matched move-agreement with Maia-1700, has a measurably lower blunder rate (≥ 1 absolute percentage point with bootstrapped 95% CI not crossing zero).
2. We have a Pareto plot with at least three points per method (BC, Maia, IQL variants).
3. Code is public, reproducible from a fresh Kaggle notebook in < 12 GPU-h.
4. Checkpoints and processed dataset are on Kaggle / HF with versioned names.
5. 4–8 page write-up with the table, the plot, an ablation, and an honest limitations section.

If by Week 7 we don't have (1), we switch the story to a negative-result paper ("offline RL with naive blunder-penalty reward does not dominate behavioral cloning at matched human-likeness; here's a careful empirical study of why"). That is also publishable at a workshop and we should not be precious about it.

---

## 9. Immediate next steps (this week)

1. Scaffold the repo (pyproject, `src/chess_rl/` package, empty modules with docstrings).
2. Wire the Kaggle MCP tool (once loaded) into `scripts/push_dataset_to_kaggle.py`.
3. Write `scripts/download_lichess.sh` and pull the 2024-01 PGN.
4. Implement `src/chess_rl/encoding.py` and unit-test on 100 known positions.
5. Stand up `notebooks/01_data_smoke.ipynb` and run it on Kaggle (no GPU yet — CPU notebook).

Each of these is a separate PR. v1 of step 1 (scaffold) is the next thing I'll ship after this plan lands.
