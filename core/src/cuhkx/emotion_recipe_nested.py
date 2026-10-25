"""Two frozen inference recipes over previously fitted emotion models."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cuhkx import emotion_fusion_nested as fusion
from cuhkx import emotion_template_nested as template

RECIPES = [
    {"name": "current", "sensor_weights": {"imu": 1.0, "skeleton": 2.0, "radar": 0.2, "visual": 1.0}},
    {"name": "historical_ratio", "sensor_weights": {"imu": 0.25, "skeleton": 1.0, "radar": 0.4, "visual": 0.5}},
]


def validate_config(config, fit_config):
    if config["recipes"] != RECIPES or config["decoders"] != fit_config["decoders"]:
        raise ValueError("only the two ordered fixed recipes and original five families are allowed")
    if len(config["decoders"]) != 5 or config["inner_folds"] != 3:
        raise ValueError("frozen recipe selection requires five families and three inner folds")
    if config["selection_view_weights"] != fit_config["selection_view_weights"]:
        raise ValueError("the complete/partial validation weights may not change")
    if not config["frozen_before_selection"] or config["new_forest_fits"] != 0:
        raise ValueError("recipe experiment must reuse its frozen source models")


def candidates(config):
    return [{**decoder, "name": f"{recipe['name']}__{decoder['name']}",
             "family": decoder["name"], "recipe": recipe["name"]}
            for recipe in config["recipes"] for decoder in config["decoders"]]


def recipe_evidence(bundle, frame, contexts, matrices, masks, target_rows, config):
    results = {}
    for recipe in config["recipes"]:
        # Only an ephemeral inference configuration changes; fitted state stays intact.
        inference = {**bundle, "config": {**bundle["config"], "sensor_weights": recipe["sensor_weights"]}}
        results[recipe["name"]] = fusion.base_evidence(inference, frame, contexts, matrices, masks, target_rows)
    return results


def validation_views(frame, evidence, config):
    choices = candidates(config)
    return pd.concat([
        template.validation_views(frame, evidence[recipe["name"]],
                                  [decoder for decoder in choices if decoder["recipe"] == recipe["name"]])
        for recipe in config["recipes"]
    ], ignore_index=True)


def choose_decoder(scores, config):
    return template.choose_decoder(scores, {**config, "decoders": candidates(config)})


def predict_rows(frame, evidence, policy):
    decisions = {}
    for _, group in frame.groupby("group", sort=False):
        decoder = template.policy_decoder(policy, group)
        decisions.update(template.decode_group(group, evidence[decoder["recipe"]], decoder))
    return pd.DataFrame([{"qa_id": row.qa_id, "prediction": decisions[row.qa_id][0],
                          "margin": decisions[row.qa_id][1]} for row in frame.itertuples()])


def check_policy(policy, config):
    if set(policy) != {"name", "complete", "partial"} or policy["name"] != template.POLICY_NAME:
        raise ValueError("recipe policy must retain the original two template scopes")
    if any(policy[scope] not in candidates(config) for scope in ("complete", "partial")):
        raise ValueError("recipe policy contains an undeclared candidate")


def assert_outputs_equal(expected, actual):
    if expected.qa_id.tolist() != actual.qa_id.tolist() or expected.prediction.tolist() != actual.prediction.tolist():
        raise ValueError("replayed prediction IDs or answers differ")
    np.testing.assert_allclose(expected.margin.astype(float), actual.margin.astype(float), rtol=1e-12, atol=1e-12)
