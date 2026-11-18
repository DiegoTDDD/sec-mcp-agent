"""MCP server `fraud-scoring`: the realtime-fraud-detection-ml model as a tool.

Tools      score_transaction
Resources  fraud://model-card   threshold, categories, cities and test-set metrics

Run over stdio:
    python -m sec_mcp_agent.fraud.server
The model directory comes from SEC_MCP_FRAUD_MODEL (default: models/fraud), built by
scripts/build_fraud_model.sh.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import ToolAnnotations

from sec_mcp_agent.fraud.model import CITY_COORDS, FraudModel, FraudScore, Transaction

DEFAULT_MODEL_DIR = Path("models/fraud")


def build_server(model: FraudModel) -> MCPServer:
    server = MCPServer(
        "fraud-scoring",
        instructions=(
            "Score card transactions with an XGBoost fraud model. Report the probability, "
            "whether it is flagged, and the top_reasons the model gives."
        ),
    )

    @server.tool(annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False))
    def score_transaction(transaction: Transaction) -> FraudScore:
        """Fraud probability for one card transaction, whether it crosses the alert threshold,
        and the features that pushed the score up or down the most."""
        try:
            return model.score(transaction)
        except ValueError as exc:
            raise ToolError(str(exc)) from exc

    @server.resource(
        "fraud://model-card",
        mime_type="application/json",
        description="What the fraud model expects and how well it scored",
    )
    def model_card() -> str:
        return json.dumps(
            {
                "threshold": model.threshold,
                "features": model.feature_cols,
                "categories": model.categories,
                "cities": list(CITY_COORDS),
                "test_metrics": model.metrics,
                "source": "https://github.com/DiegoTDDD/realtime-fraud-detection-ml",
            },
            indent=2,
        )

    return server


def main() -> None:
    directory = Path(os.environ.get("SEC_MCP_FRAUD_MODEL", DEFAULT_MODEL_DIR))
    build_server(FraudModel.load(directory)).run("stdio")


if __name__ == "__main__":
    main()
