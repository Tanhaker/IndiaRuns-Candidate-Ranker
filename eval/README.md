# Evaluation harness

There is **no live leaderboard** and only **3 submissions**. So before changing
the ranker, measure the change locally against a validation set you label by
hand. This harness reproduces the challenge's exact scoring blend:

```
composite = 0.50·NDCG@10 + 0.30·NDCG@50 + 0.15·MAP + 0.05·P@10
```

## One-time setup

```
python -m eval.make_labels --sample sample_candidates.json --out eval/labels.csv
```

Open `eval/labels.csv` and fill the **`your_tier`** column (0–4) with your own
judgement:

| tier | meaning |
|------|---------|
| 4 | perfect-fit senior AI/ML/retrieval engineer, product co, corroborated |
| 3 | strong, clearly relevant, minor gaps |
| 2 | plausible / adjacent (backend-leaning ML, strong but off-band) |
| 1 | weak — keyword overlap only, or wrong specialism |
| 0 | irrelevant **or** honeypot (impossible profile) **or** keyword-stuffer |

`suggested_tier` is a heuristic first guess to speed you up — **override it
freely**. If the suggestion were trustworthy you would not need an eval set.

## Measure

```
python -m eval.evaluate                      # scores the CURRENT ranker
python -m eval.evaluate --ranking some.csv   # scores a saved submission CSV
```

Treat the printed `COMPOSITE` as your private leaderboard. Change a weight in
`redrob_ranker/config.py`, re-run, and keep the change only if the number rises.

## Caveats (read these)

- The public sample is intentionally **stuffer-heavy** with very few strong
  candidates, so absolute numbers are low and noisy — use the harness to compare
  *versions of your ranker against each other*, not to predict your final score.
- Unlabeled rows fall back to `suggested_tier` and the tool warns you. A
  half-labeled file gives a half-trustworthy number.
- Binary metrics (MAP, P@10) count tier ≥ 3 as relevant; change `REL_THRESHOLD`
  in `metrics.py` if you label on a different cutoff.
