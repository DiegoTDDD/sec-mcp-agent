# ADR 0003: The fraud tool computes features exactly as training did, and proves it

**Status:** accepted

## Context
A model scored with features computed differently from training gives confident wrong
answers (training/serving skew). The fraud model's features include velocity
(`tx_count_last_5min`, `time_since_last_tx`) and distance from the card's home city.

## Decision
- `scripts/build_fraud_model.sh` trains the model with realtime-fraud-detection-ml's own code at
  a pinned commit, then `scripts/check_feature_parity.py` rebuilds 2,000 training rows as tool
  inputs and compares every feature and probability with the training pipeline. The eval
  workflow fails if they differ (measured: 2e-13 on features, 5e-7 on probabilities).
- Timestamps are truncated to whole seconds, because training casts them to `datetime64[s]`.
  The parity check found this: with fractional seconds, 3 of 500 sampled rows differed. The
  project's streaming scorer uses fractional seconds, so it carries the same sub-second skew; that
  is recorded as an item for that project.
- Explanations are XGBoost's exact per-feature contributions to the log-odds (TreeSHAP), so the
  agent reports the model's reasons instead of inventing them.

## Consequences
- The tool and the training pipeline agree by test, not by convention.
- The agent reports what the model says even when the model is weak. For example, a large
  purchase far from home with no card history scores low because `time_since_last_tx = 99999`
  pulls strongly towards legitimate. That matches the known impossible-travel weakness (recall
  0.65) in the fraud project, and the top reasons show it.
