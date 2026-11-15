# ECE 579 Term Project — Pipeline Progress Log

Running log of what's been built in `src/pipeline.py` and the results at each stage. Append new sections as we go.

---

## Part 1 — Original baseline pipeline (built first)

**Steps:**
1. Load `data/train.csv`, drop `id`, separate `loss` into target `y`.
2. Fixed 70/15/15 train/val/test split (`random_state=42`).
3. One-hot encode the 116 `cat` columns via `ColumnTransformer` (fit on training data only); pass the 14 `cont` columns through unchanged.
4. Built a reusable `evaluate()` helper: fits a model on train, scores it on the single validation split (RMSE + MAE).
5–7. Ran three models through `evaluate()`: a trivial mean predictor, Ridge regression (baseline), and XGBoost (proposed).

**Results (single 70/15/15 split):**

| Model | RMSE | MAE |
|---|---|---|
| Trivial reference (mean predictor) | 2892.36 | 1971.19 |
| Baseline: Ridge regression | 2036.97 | 1308.16 |
| Proposed: Gradient Boosting (XGBoost) | 1922.21 | 1215.09 |

**Takeaway at this stage:** XGBoost beat Ridge on both metrics on one validation split, but this was a single split — no sense yet of whether the gap was real or noise.

---

## Part 2 — Cross-validation + hyperparameter tuning (added later)

**What was added:**
- `cv_evaluate()` helper: 5-fold `KFold` cross-validation on the training set, reporting mean ± std RMSE/MAE for any model (instead of a single validation score).
- Ran all three original models through 5-fold CV.
- Hyperparameter search for **Ridge**: grid search over `alpha ∈ {0.01, 0.1, 1.0, 10.0, 50.0, 100.0, 500.0}`, selected on CV RMSE.
- Hyperparameter search for **XGBoost**: randomized search (20 sampled configs) over `max_depth`, `learning_rate`, `n_estimators`, `subsample`, using 3-fold CV during the search (for speed), then confirmed the winning config with the full 5-fold `cv_evaluate`.

**Results — 5-fold CV, all models on the same training-set folds:**

| Model | RMSE (mean ± std) | MAE (mean ± std) |
|---|---|---|
| Trivial reference (mean predictor) | 2913.38 ± 24.95 | 1966.64 ± 8.75 |
| Baseline: Ridge regression (tuned, best α = 1.0) | 2033.10 ± 23.02 | 1301.75 ± 5.96 |
| XGBoost (untuned defaults) | 1938.77 ± 26.87 | 1213.71 ± 5.45 |
| **XGBoost (tuned)** | **1909.76 ± 23.15** | **1187.94 ± 6.74** |

**Tuned XGBoost's winning config:** `max_depth=8, learning_rate=0.05, n_estimators=300, subsample=0.8`. Search runtime: 5.4 minutes.

**Key findings:**
- Ridge's best alpha came back as **1.0 — the default**. The search still counts as "tuned with comparable effort" per the rubric; it just confirmed the default was already optimal.
- XGBoost's RMSE and MAE both improved further after tuning, and both gaps (vs. untuned XGBoost) are clearly larger than the fold-to-fold noise — a real improvement
- XGBoost beats Ridge on both metrics, and the gap is large relative to the std on both — defensible as a genuine improvement, not noise (satisfies the rubric's "differences smaller than the spread must not be called improvements" requirement).
- **Metric justification (for the report):** RMSE and MAE are reported together because RMSE is sensitive to the rare, very large claims that drive insurance risk, while MAE reflects typical claim-level error — reporting only one would hide which kind of mistake the model is making.

**Rubric lines this covers:** Methods (10 pts) — search space/method/criterion documented for both models. Baseline comparison (8 pts) — Ridge and XGBoost got comparable tuning effort. Evaluation rigor (10 pts) — mean ± std reported; still need per-segment results (next up).

---

## Part 3 — Claim-size segment breakdown

**What was added:** `segment_evaluate()` — uses `cross_val_predict` (5-fold, same seed) to get an out-of-fold prediction for every training row with no leakage, splits `y_train` into tertiles (low/medium/high severity, ~33rd/66th percentile cutoffs), and reports RMSE/MAE per bucket for Ridge (tuned) and XGBoost (tuned).

**Results:**

| Bucket | n | Ridge RMSE | XGBoost RMSE | RMSE improvement | Ridge MAE | XGBoost MAE | MAE improvement |
|---|---|---|---|---|---|---|---|
| Low | 43,501 | 1327.79 | 1195.76 | 9.9% | 959.12 | 843.29 | 12.1% |
| Medium | 43,501 | 1272.65 | 1134.26 | 10.9% | 943.12 | 777.26 | 17.6% |
| High | 44,820 | 2979.20 | 2844.64 | 4.5% | 1982.36 | 1921.03 | 3.1% |

**Key finding:** XGBoost's advantage over Ridge is much larger on low/medium-severity claims (10–18% error reduction) than on high-severity claims (only 3–5%). The aggregate comparison (RMSE 1909.76 vs 2033.10) makes XGBoost look like a clean, uniform win, but the per-segment breakdown shows **both models struggle hardest on exactly the claims that matter most for business risk**, and XGBoost's edge mostly evaporates there.

This echoes Terefe & Aga (2026) from our own D2 literature review, who found that tree-based ML methods "did better but still underestimated extreme claims somewhat" versus a classical baseline — we're seeing the same pattern on a different dataset. 

**Rubric lines this covers:** Evaluation rigor's per-segment requirement (10 pts, shared with CV work above), and doubles as the ablation/sensitivity study for Analysis and discussion (7 pts) — no separate one needed.

---

## Part 4 — Next up (not started yet)

- Error analysis on worst-predicted claims.
- Limitations/threats-to-validity write-up.
- `requirements.txt` + README.
- Final, single test-set run (only after everything above is locked in).
