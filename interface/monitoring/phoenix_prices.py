"""Copy the project's LLM tariffs into Phoenix's model pricing table.

Phoenix ignores the ``llm.cost.*`` span attributes: it prices LLM spans
itself from ``llm.model_name`` and the token counts, using its own table. That
table has no Mistral-hosted model, so traces show a $0 total cost until each
model gets a custom entry. This script creates or updates one entry per model
of ``configured_llm_cost_rates()``. Phoenix prices spans when they arrive, so
spans received before the entry exists stay at $0.

Usage (prod): docker compose exec api python -m interface.monitoring.phoenix_prices
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.request
from typing import Any

from interface.backend.llm_providers import configured_llm_cost_rates

LIST_QUERY = """
{ generativeModels(first: 1000) { edges { node {
  id name kind namePattern tokenPrices { tokenType kind costPerMillionTokens }
} } } }
"""
CREATE_MUTATION = """
mutation($input: CreateModelMutationInput!) { createModel(input: $input) { __typename } }
"""
UPDATE_MUTATION = """
mutation($input: UpdateModelMutationInput!) { updateModel(input: $input) { __typename } }
"""


def phoenix_base_url() -> str:
    endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://phoenix:6006/v1/traces")
    return endpoint.rstrip("/").removesuffix("/v1/traces")


def graphql(base_url: str, query: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
    headers = {"content-type": "application/json"}
    api_key = os.getenv("PHOENIX_API_KEY", "").strip()
    if api_key:
        headers["authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        f"{base_url}/graphql",
        data=json.dumps({"query": query, "variables": variables or {}}).encode(),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.load(response)
    if payload.get("errors"):
        raise RuntimeError(f"Phoenix GraphQL: {payload['errors']}")
    return payload["data"]


def model_input(name: str, prompt_rate: float, completion_rate: float) -> dict[str, Any]:
    return {
        "name": name,
        "provider": "mistral",
        "namePattern": f"^{re.escape(name)}$",
        "costs": [
            {"tokenType": "input", "kind": "PROMPT", "costPerMillionTokens": prompt_rate},
            {"tokenType": "output", "kind": "COMPLETION", "costPerMillionTokens": completion_rate},
        ],
    }


def current_prices(node: dict[str, Any]) -> dict[str, float]:
    return {price["tokenType"]: price["costPerMillionTokens"] for price in node["tokenPrices"]}


def sync_prices(base_url: str, rates: dict[str, tuple[float, float]]) -> list[str]:
    """Create or update a custom Phoenix model per tariff; returns the actions taken."""
    edges = graphql(base_url, LIST_QUERY)["generativeModels"]["edges"]
    custom_by_name = {
        edge["node"]["name"]: edge["node"] for edge in edges if edge["node"]["kind"] == "CUSTOM"
    }
    actions = []
    for name, (prompt_rate, completion_rate) in sorted(rates.items()):
        wanted = model_input(name, prompt_rate, completion_rate)
        existing = custom_by_name.get(name)
        if existing is None:
            graphql(base_url, CREATE_MUTATION, {"input": wanted})
            actions.append(f"created {name}")
            continue
        unchanged = (
            existing["namePattern"] == wanted["namePattern"]
            and current_prices(existing) == {"input": prompt_rate, "output": completion_rate}
        )
        if unchanged:
            actions.append(f"unchanged {name}")
            continue
        graphql(base_url, UPDATE_MUTATION, {"input": {"id": existing["id"], **wanted}})
        actions.append(f"updated {name}")
    return actions


def main() -> int:
    for action in sync_prices(phoenix_base_url(), configured_llm_cost_rates()):
        print(action)
    return 0


if __name__ == "__main__":
    sys.exit(main())
