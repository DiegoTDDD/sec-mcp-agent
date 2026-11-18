from __future__ import annotations

import json
import random
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import pandas as pd
import pytest
from xgboost import XGBClassifier

from sec_mcp_agent.filings.corpus import Chunk, make_chunks
from sec_mcp_agent.filings.search import FilingsIndex
from sec_mcp_agent.fraud.model import CITY_COORDS, FraudModel

FIXTURES = Path(__file__).parent / "fixtures"
CATEGORIES = ["electronics", "groceries", "restaurant", "travel"]


def fixture_chunks() -> list[Chunk]:
    data = json.loads((FIXTURES / "filings.json").read_text(encoding="utf-8"))
    chunks: list[Chunk] = []
    for f in data:
        chunks += make_chunks(
            f["ticker"], f["company"], f["filing_date"], f["section"], f["text"], max_chars=400
        )
    return chunks


@pytest.fixture(scope="session")
def chunks() -> list[Chunk]:
    return fixture_chunks()


@pytest.fixture(scope="session")
def index(chunks: list[Chunk]) -> FilingsIndex:
    return FilingsIndex(chunks)


@pytest.fixture(scope="session")
def fraud_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A small model with the same artifact layout as realtime-fraud-detection-ml.

    Fraud in the synthetic data is either a large amount or a purchase far from home, so tests
    can check the model and its explanations react to the right features.
    """
    rng = random.Random(0)
    cities = list(CITY_COORDS)
    home = {f"card_{i:04d}": rng.choice(cities) for i in range(50)}
    rows = []
    start = datetime(2026, 1, 1)
    for i in range(3000):
        card = rng.choice(list(home))
        kind = rng.choices(["legit", "high_amount", "travel"], weights=[80, 10, 10])[0]
        fraud = kind != "legit"
        city = (
            rng.choice([c for c in cities if c != home[card]]) if kind == "travel" else home[card]
        )
        rows.append(
            {
                "amount": rng.uniform(2000, 5000) if kind == "high_amount" else rng.uniform(5, 400),
                "distance_km_from_home": 0.0 if city == home[card] else 1500.0,
                "tx_count_last_5min": 0,
                "time_since_last_tx": 99999.0,
                "hour_of_day": (start + timedelta(minutes=i)).hour,
                "category": rng.choice(CATEGORIES),
                "is_fraud": int(fraud),
            }
        )
    df = pd.DataFrame(rows)
    dummies = pd.get_dummies(df["category"], prefix="category").astype(int)
    features = [
        "amount",
        "distance_km_from_home",
        "tx_count_last_5min",
        "time_since_last_tx",
        "hour_of_day",
        *dummies.columns,
    ]
    x = pd.concat([df, dummies], axis=1)[features]
    model = XGBClassifier(n_estimators=40, max_depth=3, random_state=0).fit(x, df["is_fraud"])
    out = tmp_path_factory.mktemp("fraud")
    joblib.dump({"model": model, "feature_cols": features, "threshold": 0.5}, out / "model.joblib")
    (out / "card_home_city.json").write_text(json.dumps(home), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps({"pr_auc": 0.99}), encoding="utf-8")
    return out


@pytest.fixture(scope="session")
def fraud_model(fraud_dir: Path) -> FraudModel:
    return FraudModel.load(fraud_dir)
