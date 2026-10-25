"""Nested emotion selection with three-question partial-core validation views.

Synthetic views relax option-core constraints while keeping every question and
the original unlabeled sensor/context evidence. They simulate the constraint
graph, not the vocabulary shift in actual partial-core questions.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from cuhkx import emotion_fusion_nested as fusion
from cuhkx import emotion_nested as legacy

POLICY_NAME = "template_routed"


def complete_core(group):
    return len(group) == 3 and len(set.intersection(*(set(x) for x in group.options))) == 3


def policy_decoder(policy, group):
    if policy.get("name") != POLICY_NAME:
        return policy
    return policy["complete" if complete_core(group) else "partial"]


def relaxed_assignments(group, pair):
    """Permit the third question's distractor without removing that question."""
    if not complete_core(group) or len(pair) != 2 or len(set(pair)) != 2:
        raise ValueError("partial-core proxy requires a full common-core triplet and two rows")
    if not set(pair) <= set(range(3)):
        raise ValueError("proxy pair indices must belong to this triplet")
    options = list(group.options)
    core = set.intersection(*(set(x) for x in options))
    allowed = [core if index in pair else set(words) for index, words in enumerate(options)]
    assignments = [words for words in itertools.product(*options)
                   if len(set(words)) == 3
                   and all(word in valid for word, valid in zip(words, allowed, strict=True))]
    if not assignments:
        raise ValueError("partial-core proxy has no valid assignment")
    return assignments


def decode_group(group, evidence, decoder, *, assignments=None):
    decoder = policy_decoder(decoder, group)
    adapted = fusion.adapted_evidence({qa_id: evidence[qa_id] for qa_id in group.qa_id}, decoder)
    if assignments is None:
        return legacy.decode_group(group, adapted, decoder)
    rows = list(group.itertuples())
    values = []
    for words in assignments:
        if len(words) != len(rows):
            raise ValueError("proxy assignment lost a question")
        value = 0.0
        for row, word in zip(rows, words, strict=True):
            item = adapted[row.qa_id][word]
            semantic = decoder["lexical"] * item["lexical"] + decoder["context"] * item["context"]
            value += semantic[row.position] + decoder["sensor"] * item["sensor"]
        values.append(value)
    order = np.argsort(-np.asarray(values), kind="stable")
    words = assignments[int(order[0])]
    margin = float(values[order[0]] - values[order[1]]) if len(order) > 1 else 0.0
    return {row.qa_id: (row.letters[row.options.index(word)], margin)
            for row, word in zip(rows, words, strict=True)}


def predict_rows(frame, evidence, policy):
    decisions = {}
    for _, group in frame.groupby("group", sort=False):
        decisions.update(decode_group(group, evidence, policy))
    return pd.DataFrame([{"qa_id": row.qa_id, "prediction": decisions[row.qa_id][0],
                          "margin": decisions[row.qa_id][1]} for row in frame.itertuples()])


def actual_partial_roles(group):
    options = [set(words) for words in group.options]
    pairs = [(i, j) for i, j in itertools.combinations(range(len(group)), 2)
             if len(options[i] & options[j]) == 3]
    paired = set(pairs[0]) if len(pairs) == 1 else set()
    return ["pair" if index in paired else "singleton" for index in range(len(group))]


def validation_views(frame, evidence, decoders):
    records = []
    for group_id, group in frame.groupby("group", sort=False):
        if complete_core(group):
            views = [("actual_complete", ["complete"] * 3, None)]
            for pair in itertools.combinations(range(3), 2):
                roles = ["pair" if index in pair else "singleton" for index in range(3)]
                views.append((f"relaxed_pair_{pair[0]}_{pair[1]}", roles,
                              relaxed_assignments(group, pair)))
        else:
            views = [("actual_partial", actual_partial_roles(group), None)]
        for view_id, roles, assignments in views:
            for candidate in decoders:
                if candidate.get("name") == POLICY_NAME:
                    decoder = candidate["complete" if view_id == "actual_complete" else "partial"]
                else:
                    decoder = candidate
                decisions = decode_group(group, evidence, decoder, assignments=assignments)
                for row, role in zip(group.itertuples(), roles, strict=True):
                    records.append({"qa_id": row.qa_id, "group": group_id, "subject": row.subject,
                                    "view": role, "view_id": view_id, "visible_questions": len(group),
                                    "decoder": candidate["name"], "correct": decisions[row.qa_id][0] == row.answer})
    return pd.DataFrame(records)


def choose_decoder(scores, config):
    results, selected = [], {}
    for scope, weights in config["selection_view_weights"].items():
        local_results = []
        for order, decoder in enumerate(config["decoders"]):
            rows = scores.loc[scores.decoder.eq(decoder["name"])]
            by_view = rows.groupby("view").correct.mean().to_dict()
            subjects = rows.groupby(["subject", "view"]).correct.mean().unstack()
            if not set(weights) <= set(by_view):
                raise ValueError("inner validation lacks a required template view")
            if subjects[list(weights)].isna().any().any():
                raise ValueError("inner subject lacks a required template view")
            weighted = sum(weights[view] * by_view[view] for view in weights)
            worst = float(sum(weights[view] * subjects[view] for view in weights).min())
            record = {"scope": scope, "name": decoder["name"], "weighted_accuracy": float(weighted),
                      "worst_subject_weighted_accuracy": worst, "by_view": by_view, "order": order}
            local_results.append(record)
            results.append(record)
        best = max(local_results, key=lambda row: (
            row["weighted_accuracy"], row["worst_subject_weighted_accuracy"], -row["order"]))
        selected[scope] = next(decoder for decoder in config["decoders"] if decoder["name"] == best["name"])
    return {"name": POLICY_NAME, **selected}, results
