# Learning from Losses: Offline RL on Human Blunders for Human-Aligned Chess Engines

**Author:** [Your Name] · **Draft:** v0.1

## 1. Motivation

Modern chess engines (Stockfish, Leela) optimize for *strength*, while Maia (McIlroy-Young et al., KDD 2020) models *typical human play* at a target Elo. Neither is what an improving human actually wants. A human student wants a partner that plays like them — but who has *studied their own mistakes*. We ask: can we train a policy that preserves human-likeness at a target Elo while systematically avoiding the blunder-class moves that humans at that Elo make?

## 2. Research Question

**RQ.** Given a corpus of human games with engine-annotated blunders, can offline RL produce a policy that, at a fixed human-likeness budget (move-agreement with target-Elo humans), achieves a measurably lower blunder rate than (a) supervised behavioral cloning on the same data and (b) Maia at matched Elo?

## 3. Data

- **Lichess open database** (free, ~100GB/month, CC0). Subsample 5M positions from games rated 1400–1800.
- Each game already includes per-move Stockfish evaluations and `?`/`??` tags.
- State: 8×8×12 piece-plane tensor + side-to-move + castling/en-passant flags.
- Reward shaping: `r = -Δeval(move)` clipped to [−3, +1] pawns, with extra −1 penalty on tagged `??` moves.

## 4. Method

1. **Baselines.** (B1) Behavioral cloning on all moves. (B2) BC filtered to non-blunder moves only. (B3) Maia-1500 / Maia-1700 public checkpoints.
2. **Proposed.** Offline RL via Conservative Q-Learning (CQL) and Implicit Q-Learning (IQL) using `d3rlpy`. Policy: 6-block ResNet over the board tensor (~1.5M params). Train on the same 5M-position dataset with the reward above.
3. **Variant.** Replay-weighted training that upsamples positions immediately preceding tagged blunders, to focus learning on tactically critical states.

## 5. Evaluation

- **Blunder rate** on a 50k-position held-out set: fraction of moves Stockfish (depth 18) classifies as `??`.
- **Human-likeness:** top-1 move-agreement with held-out games at the target Elo bucket.
- **Playing strength:** 1000-game self-play tournaments vs Maia-1500/1700 and vs Stockfish nerfed to matched Elo via skill-level + depth limits.
- **Headline plot:** Pareto curve of human-likeness vs blunder rate; we expect our method to dominate BC and Maia on this frontier.

## 6. Compute Budget

Single Colab/Kaggle T4 (16 GB).
- Data prep: ~6 h (one-time) on CPU.
- Each training run: ~8 h. Plan for ~10 runs across CQL/IQL × hyperparams × Elo buckets.
- Evaluation tournaments: Stockfish + python-chess on CPU, ~12 h total.
- **Total: under Kaggle's 30 GPU-h/week quota for ~3 weeks.**

## 7. Timeline (10 weeks)

| Week | Milestone |
|---|---|
| 1 | Lichess data pipeline, board encoder, Stockfish eval cache |
| 2 | BC baseline trained + held-out blunder-rate metric working |
| 3–4 | CQL implementation + first training runs |
| 5 | IQL variant + replay weighting |
| 6 | Maia checkpoint integration, head-to-head tournament harness |
| 7 | Full evaluation sweep, Pareto curves |
| 8 | Ablations (reward shaping, dataset size, architecture) |
| 9 | Writing |
| 10 | Polish, figures, submission |

## 8. Target Venues

- **Primary:** NeurIPS Workshop on AI for Games (2026), or ML for Creativity & Design.
- **Secondary:** ICLR Tiny Papers track, or arXiv preprint with code release.
- **Stretch:** AAAI Student Abstract.

## 9. Risks & Mitigations

- *Risk:* CQL underperforms BC. *Mitigation:* IQL is the fallback; even a negative result on CQL is publishable as a finding if framed honestly.
- *Risk:* Maia checkpoints already dominate the Pareto frontier. *Mitigation:* The contribution is the *blunder-aware* axis, not raw strength — pre-register the metric.
- *Risk:* Compute overrun. *Mitigation:* Subsample to 1M positions; the method should still be demonstrable.

## 10. Deliverables

- Open-source repo (PyTorch + d3rlpy) with reproducible training scripts.
- Released model checkpoints at 3 Elo buckets.
- 4–8 page workshop paper.
