#!/usr/bin/env bash
# Train the fraud model exactly as realtime-fraud-detection-ml does, at a pinned commit, copy
# the artifacts to models/fraud/ and check this project computes the same features.
#
#   bash scripts/build_fraud_model.sh
set -euo pipefail

REPO=https://github.com/DiegoTDDD/realtime-fraud-detection-ml
COMMIT=3ebb8bba1cce094070b6411a14442bcab8b16a0e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

git clone --quiet "$REPO" "$WORK"
git -C "$WORK" checkout --quiet "$COMMIT"
(
  cd "$WORK"
  mkdir -p data docs/screenshots
  python training/generate_training_data.py
  python training/train_model.py
)
mkdir -p "$ROOT/models/fraud"
cp "$WORK"/training/{model.joblib,card_home_city.json,metrics.json} "$ROOT/models/fraud/"
echo "$REPO@$COMMIT" > "$ROOT/models/fraud/SOURCE"
python "$ROOT/scripts/check_feature_parity.py" "$WORK"
