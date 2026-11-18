"""Score one card transaction with the XGBoost model from realtime-fraud-detection-ml.

Features are computed exactly as that project's streaming scorer computes them
(scoring/score_stream.py), so the MCP tool and the Spark job agree on every input:

    distance_km_from_home  great-circle km between the transaction city and the card's home city
    tx_count_last_5min     earlier transactions of the card in the 300 s before this one
    time_since_last_tx     seconds since the card's previous transaction (99999 if none)
    hour_of_day, amount, one-hot category

The explanation is XGBoost's exact per-feature contribution to the log-odds (TreeSHAP),
so "why was this flagged" comes from the model, not from the language model.
"""

from __future__ import annotations

import json
import math
import unicodedata
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from pydantic import BaseModel, Field

# Same coordinates as realtime-fraud-detection-ml/training/train_model.py.
CITY_COORDS = {
    "Sao Paulo": (-23.5505, -46.6333),
    "Rio de Janeiro": (-22.9068, -43.1729),
    "Belo Horizonte": (-19.9167, -43.9345),
    "Curitiba": (-25.4284, -49.2733),
    "Porto Alegre": (-30.0346, -51.2177),
    "Salvador": (-12.9777, -38.5016),
    "Recife": (-8.0476, -34.8770),
    "Fortaleza": (-3.7172, -38.5433),
    "Brasilia": (-15.7939, -47.8828),
    "Manaus": (-3.1190, -60.0217),
}
VELOCITY_WINDOW_SECONDS = 300
NO_PREVIOUS_TX = 99999.0


def canonical_city(name: str) -> str:
    """'São Paulo', 'sao paulo' and 'SAO PAULO' all mean 'Sao Paulo'; unknown names pass through."""
    folded = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower().strip()
    return next((c for c in CITY_COORDS if c.lower() == folded), name.strip())


def haversine_km(city_a: str, city_b: str) -> float:
    if city_a == city_b:
        return 0.0
    lat1, lon1, lat2, lon2 = map(math.radians, (*CITY_COORDS[city_a], *CITY_COORDS[city_b]))
    a = (
        math.sin((lat2 - lat1) / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * 6371 * math.asin(math.sqrt(a))


class Transaction(BaseModel):
    card_id: str = Field(description="Card identifier, e.g. card_0042")
    amount: float = Field(gt=0, description="Amount in BRL")
    category: str = Field(description="Merchant category, e.g. electronics")
    city: str = Field(description="City where the transaction happened")
    timestamp: datetime = Field(description="When it happened (ISO 8601)")
    previous_timestamps: list[datetime] = Field(
        default_factory=list,
        description="Times of the card's earlier transactions, if known (for velocity features)",
    )


class Contribution(BaseModel):
    feature: str
    value: float
    log_odds: float = Field(description="Push towards fraud (+) or legitimate (-)")


class FraudScore(BaseModel):
    probability: float
    flagged: bool
    threshold: float
    home_city: str | None
    features: dict[str, float]
    top_reasons: list[Contribution]


class FraudModel:
    def __init__(
        self,
        model: xgb.XGBClassifier,
        feature_cols: list[str],
        threshold: float,
        home_city: dict[str, str],
        metrics: dict | None = None,
    ) -> None:
        self.model = model
        self.feature_cols = feature_cols
        self.threshold = threshold
        self.home_city = home_city
        self.metrics = metrics or {}
        self.categories = sorted(
            c.removeprefix("category_") for c in feature_cols if c.startswith("category_")
        )

    @classmethod
    def load(cls, directory: str | Path) -> FraudModel:
        directory = Path(directory)
        artifact = joblib.load(directory / "model.joblib")
        home_city = json.loads((directory / "card_home_city.json").read_text(encoding="utf-8"))
        metrics_path = directory / "metrics.json"
        metrics = (
            json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
        )
        return cls(
            artifact["model"],
            artifact["feature_cols"],
            float(artifact["threshold"]),
            home_city,
            metrics,
        )

    def features(self, tx: Transaction) -> dict[str, float]:
        if tx.city not in CITY_COORDS:
            raise ValueError(f"Unknown city {tx.city!r}. Known: {', '.join(CITY_COORDS)}")
        if tx.category not in self.categories:
            raise ValueError(
                f"Unknown category {tx.category!r}. Known: {', '.join(self.categories)}"
            )
        home = self.home_city.get(tx.card_id)
        # Whole seconds, as in training (timestamps cast to datetime64[s]); see ADR 0003.
        ts = math.floor(tx.timestamp.timestamp())
        earlier = sorted(
            s for s in (math.floor(t.timestamp()) for t in tx.previous_timestamps) if s <= ts
        )
        row = {
            "amount": tx.amount,
            "distance_km_from_home": haversine_km(tx.city, home) if home else 0.0,
            "tx_count_last_5min": float(sum(ts - t <= VELOCITY_WINDOW_SECONDS for t in earlier)),
            "time_since_last_tx": ts - earlier[-1] if earlier else NO_PREVIOUS_TX,
            "hour_of_day": float(tx.timestamp.hour),
        }
        for col in self.feature_cols:
            if col.startswith("category_"):
                row[col] = float(col == f"category_{tx.category}")
        return {col: float(row[col]) for col in self.feature_cols}

    def score(self, tx: Transaction, top: int = 3) -> FraudScore:
        feats = self.features(tx)
        frame = pd.DataFrame([feats], columns=self.feature_cols)
        probability = float(self.model.predict_proba(frame)[0, 1])
        contribs = self.model.get_booster().predict(xgb.DMatrix(frame), pred_contribs=True)[0]
        order = np.argsort(-np.abs(contribs[:-1]))[:top]  # last column is the bias
        reasons = [
            Contribution(
                feature=self.feature_cols[i],
                value=feats[self.feature_cols[i]],
                log_odds=round(float(contribs[i]), 4),
            )
            for i in order
        ]
        return FraudScore(
            probability=round(probability, 6),
            flagged=probability >= self.threshold,
            threshold=round(self.threshold, 6),
            home_city=self.home_city.get(tx.card_id),
            features=feats,
            top_reasons=reasons,
        )
