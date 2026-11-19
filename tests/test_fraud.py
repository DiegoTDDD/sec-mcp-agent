"""Fraud features, scoring with explanations, and the fraud-scoring MCP server."""

from datetime import datetime, timedelta

import pytest
from mcp.client import Client

from sec_mcp_agent.fraud.model import NO_PREVIOUS_TX, Transaction, haversine_km
from sec_mcp_agent.fraud.server import build_server

pytestmark = pytest.mark.anyio
T0 = datetime(2026, 3, 1, 14, 30, 0)


def tx(fraud_model, **overrides) -> Transaction:
    card = "card_0000"
    fields = {
        "card_id": card,
        "amount": 50.0,
        "category": "groceries",
        "city": fraud_model.home_city[card],
        "timestamp": T0,
    }
    return Transaction(**(fields | overrides))


def test_haversine_matches_known_distances():
    assert haversine_km("Sao Paulo", "Sao Paulo") == 0
    assert haversine_km("Sao Paulo", "Rio de Janeiro") == pytest.approx(361, abs=5)
    assert haversine_km("Sao Paulo", "Manaus") == pytest.approx(2689, abs=10)


def test_velocity_features_follow_the_training_definition(fraud_model):
    prev = [T0 - timedelta(seconds=s) for s in (400, 300, 120, 10)] + [T0 + timedelta(seconds=5)]
    f = fraud_model.features(tx(fraud_model, previous_timestamps=prev))
    assert f["tx_count_last_5min"] == 3  # 300 s inclusive, 400 s out, future ignored
    assert f["time_since_last_tx"] == 10
    assert fraud_model.features(tx(fraud_model))["time_since_last_tx"] == NO_PREVIOUS_TX


def test_timestamps_are_truncated_to_whole_seconds_like_training(fraud_model):
    f = fraud_model.features(
        tx(
            fraud_model,
            timestamp=T0 + timedelta(milliseconds=900),
            previous_timestamps=[T0 - timedelta(seconds=4, milliseconds=300)],
        )
    )
    assert f["time_since_last_tx"] == 5  # floor(t) - floor(prev) = 0 - (-5)


def test_distance_and_category_features(fraud_model):
    home = fraud_model.home_city["card_0000"]
    away = "Manaus" if home != "Manaus" else "Recife"
    f = fraud_model.features(tx(fraud_model, city=away, category="travel"))
    assert f["distance_km_from_home"] == pytest.approx(haversine_km(away, home))
    assert f["category_travel"] == 1 and f["category_groceries"] == 0
    assert (
        fraud_model.features(tx(fraud_model, card_id="card_unknown"))["distance_km_from_home"] == 0
    )


def test_unknown_city_or_category_is_rejected(fraud_model):
    with pytest.raises(ValueError, match="Unknown city"):
        fraud_model.features(tx(fraud_model, city="Lisbon"))
    with pytest.raises(ValueError, match="Unknown category"):
        fraud_model.features(tx(fraud_model, category="casino"))


def test_score_flags_the_pattern_and_explains_it_with_the_right_features(fraud_model):
    home = fraud_model.home_city["card_0000"]
    away = "Manaus" if home != "Manaus" else "Recife"
    risky = fraud_model.score(tx(fraud_model, amount=4500, city=away))
    normal = fraud_model.score(tx(fraud_model))
    assert risky.flagged and not normal.flagged
    assert risky.probability > 0.9 > 0.1 > normal.probability
    reasons = {r.feature: r.log_odds for r in risky.top_reasons}
    assert reasons.get("amount", 0) > 0 and reasons.get("distance_km_from_home", 0) > 0


async def test_server_scores_through_mcp_with_flat_arguments(fraud_model):
    async with Client(build_server(fraud_model)) as client:
        tools = (await client.list_tools()).tools
        assert [t.name for t in tools] == ["score_transaction"]
        props = tools[0].input_schema["properties"]
        assert set(props) == {
            "card_id",
            "amount",
            "category",
            "city",
            "timestamp",
            "previous_timestamps",
        }
        assert "$defs" not in tools[0].input_schema  # nothing nested for the model to copy
        args = {
            "card_id": "card_0000",
            "amount": 30,
            "category": "groceries",
            "city": fraud_model.home_city["card_0000"],
            "timestamp": "2026-03-01T14:30:00",
        }
        result = await client.call_tool("score_transaction", args)
        assert not result.is_error
        score = result.structured_content
        assert set(score) >= {"probability", "flagged", "threshold", "top_reasons"}
        assert score["flagged"] is False


async def test_city_and_category_spelling_is_forgiven(fraud_model):
    async with Client(build_server(fraud_model)) as client:
        args = {
            "card_id": "card_0001",
            "amount": 30,
            "category": "Groceries",
            "city": "SÃO PAULO",
            "timestamp": "2026-03-01T14:30:00",
        }
        result = await client.call_tool("score_transaction", args)
        assert not result.is_error


async def test_bad_transaction_is_a_readable_tool_error(fraud_model):
    async with Client(build_server(fraud_model)) as client:
        args = {
            "card_id": "card_0000",
            "amount": 30,
            "category": "casino",
            "city": "Recife",
            "timestamp": "2026-03-01T14:30:00",
        }
        result = await client.call_tool("score_transaction", args)
        assert result.is_error and "Unknown category" in result.content[0].text
        bad_city = await client.call_tool(
            "score_transaction", dict(args, category="fuel", city="Lisbon")
        )
        assert bad_city.is_error and "Unknown city" in bad_city.content[0].text
        negative = dict(args, category="groceries", amount=-5)
        assert (await client.call_tool("score_transaction", negative)).is_error


async def test_model_card_resource(fraud_model):
    async with Client(build_server(fraud_model)) as client:
        res = await client.read_resource("fraud://model-card")
        assert '"threshold": 0.5' in res.contents[0].text
        assert '"pr_auc": 0.99' in res.contents[0].text


def test_utc_and_naive_timestamps_are_the_same_transaction(fraud_model):
    aware = tx(fraud_model, timestamp="2026-03-01T14:30:00Z")
    naive = tx(fraud_model, timestamp="2026-03-01T14:30:00")
    offset = tx(fraud_model, timestamp="2026-03-01T11:30:00-03:00")
    assert aware == naive == offset
    assert fraud_model.features(aware) == fraud_model.features(naive)


async def test_previous_times_as_one_comma_separated_string(fraud_model):
    async with Client(build_server(fraud_model)) as client:
        args = {
            "card_id": "card_0000",
            "amount": 3.5,
            "category": "groceries",
            "city": fraud_model.home_city["card_0000"],
            "timestamp": "2026-03-01T10:01:25",
            "previous_timestamps": "2026-03-01T10:00:00, 2026-03-01T10:01:02",
        }
        res = await client.call_tool("score_transaction", args)
        assert not res.is_error
        assert res.structured_content["features"]["tx_count_last_5min"] == 2
