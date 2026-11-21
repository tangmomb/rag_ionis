from __future__ import annotations

import unittest
from unittest.mock import patch

from interface.monitoring import phoenix_prices


def model_node(node_id: str, name: str, kind: str, prompt: float, completion: float) -> dict:
    return {
        "node": {
            "id": node_id,
            "name": name,
            "kind": kind,
            "namePattern": f"^{name}$",
            "tokenPrices": [
                {"tokenType": "input", "kind": "PROMPT", "costPerMillionTokens": prompt},
                {"tokenType": "output", "kind": "COMPLETION", "costPerMillionTokens": completion},
            ],
        }
    }


class PhoenixPricesTests(unittest.TestCase):
    def test_creates_missing_updates_changed_and_keeps_identical_models(self) -> None:
        listing = {
            "generativeModels": {
                "edges": [
                    # Pattern escaped like the script writes it.
                    {"node": {**model_node("1", "mistral-small-latest", "CUSTOM", 0.15, 0.6)["node"],
                              "namePattern": r"^mistral\-small\-latest$"}},
                    model_node("2", "mistral-medium-latest", "CUSTOM", 1.0, 1.0),
                    # A built-in entry with the same name is never modified.
                    model_node("3", "deepseek-v4-1-flash-1017", "BUILT_IN", 9.0, 9.0),
                ]
            }
        }
        calls = []

        def fake_graphql(base_url, query, variables=None):
            calls.append((query, variables))
            return listing if "generativeModels" in query else {}

        rates = {
            "mistral-small-latest": (0.15, 0.6),
            "mistral-medium-latest": (1.5, 7.5),
            "deepseek-v4-1-flash-1017": (0.3, 1.2),
        }
        with patch.object(phoenix_prices, "graphql", side_effect=fake_graphql):
            actions = phoenix_prices.sync_prices("http://phoenix:6006", rates)

        self.assertEqual(actions, [
            "created deepseek-v4-1-flash-1017",
            "updated mistral-medium-latest",
            "unchanged mistral-small-latest",
        ])
        mutations = [variables["input"] for query, variables in calls if variables]
        self.assertEqual(mutations[0]["name"], "deepseek-v4-1-flash-1017")
        self.assertNotIn("id", mutations[0])
        self.assertEqual(mutations[0]["costs"][0]["costPerMillionTokens"], 0.3)
        self.assertEqual(mutations[1]["id"], "2")
        self.assertEqual(mutations[1]["costs"][1]["costPerMillionTokens"], 7.5)

    def test_base_url_is_derived_from_the_collector_endpoint(self) -> None:
        with patch.dict("os.environ", {"PHOENIX_COLLECTOR_ENDPOINT": "http://phoenix:6006/v1/traces"}):
            self.assertEqual(phoenix_prices.phoenix_base_url(), "http://phoenix:6006")


if __name__ == "__main__":
    unittest.main()
