"""Check the MCP tool computes the same features and scores as the training pipeline.

Rebuilds the training features with realtime-fraud-detection-ml's own code, then, for a
sample of transactions, rebuilds each one as the MCP tool would receive it (the card's
earlier timestamps included) and compares features and fraud probabilities.

    python scripts/check_feature_parity.py <path to a realtime-fraud-detection-ml checkout>
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sec_mcp_agent.fraud.model import FraudModel, Transaction  # noqa: E402

SAMPLE = 2000


def main() -> None:
    fraud_repo = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(fraud_repo / "training"))
    import train_model  # realtime-fraud-detection-ml/training/train_model.py

    model = FraudModel.load(ROOT / "models" / "fraud")
    raw = pd.read_parquet(fraud_repo / "data" / "training_raw.parquet")
    feats, cols, _ = train_model.build_features(raw.copy())
    assert cols == model.feature_cols, f"feature columns differ: {cols} vs {model.feature_cols}"
    history = (
        feats.sort_values(["card_id", "timestamp"]).groupby("card_id")["timestamp"].apply(list)
    )

    sample = feats.sample(SAMPLE, random_state=0)
    expected = model.model.predict_proba(sample[cols].astype(float))[:, 1]
    worst_feature, worst_prob = 0.0, 0.0
    for (_, row), p_expected in zip(sample.iterrows(), expected, strict=True):
        earlier = [t.to_pydatetime() for t in history[row.card_id] if t < row.timestamp]
        tx = Transaction(
            card_id=row.card_id,
            amount=row.amount,
            category=row.category,
            city=row.city,
            timestamp=row.timestamp.to_pydatetime(),
            previous_timestamps=earlier,
        )
        got = model.features(tx)
        worst_feature = max(worst_feature, *(abs(got[c] - float(row[c])) for c in cols))
        worst_prob = max(worst_prob, abs(model.score(tx).probability - float(p_expected)))
    print(
        f"{SAMPLE} transactions: max feature difference {worst_feature:.2e}, "
        f"max probability difference {worst_prob:.2e}"
    )
    if worst_feature > 1e-6 or worst_prob > 1e-5:
        raise SystemExit("training/serving skew: the MCP tool disagrees with the training pipeline")


if __name__ == "__main__":
    main()
