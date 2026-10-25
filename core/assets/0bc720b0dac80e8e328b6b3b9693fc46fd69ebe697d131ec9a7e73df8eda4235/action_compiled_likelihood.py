"""Compile answer constraints once, then score them with varying visual probabilities."""

from __future__ import annotations

import math

from cuhkx.action_partial_likelihood import negative_clause_probability
from cuhkx.metadata import option_atoms
from cuhkx.submission import available_letters


def compile_state(group, state):
    positive, negative, clauses = set(state.positives), set(), []
    answers = state.answer_map()
    for _, row in group.iterrows():
        if row.category not in answers:
            continue
        for letter in available_letters(row):
            if letter in answers[row.category]:
                continue
            atoms = set(option_atoms(row[letter]))
            if row.category in {"single", "multi"}:
                if len(atoms) != 1:
                    raise ValueError("single/multi action options must be atomic")
                negative.update(atoms)
            elif row.category == "combination":
                clauses.append(atoms)
    return {"positive": sorted(positive), "negative": sorted(negative),
            "clauses": [sorted(clause - positive) for clause in clauses if not clause & negative],
            "contradictory": bool(positive & negative)}


def score_compiled(event, probabilities):
    if event["contradictory"]:
        return -math.inf
    p = {key: min(max(float(value), 1e-5), 1 - 1e-5) for key, value in probabilities.items()}
    residual = negative_clause_probability(event["clauses"], p)
    if residual <= 0:
        return -math.inf
    return (sum(math.log(p[atom]) for atom in event["positive"])
            + sum(math.log1p(-p[atom]) for atom in event["negative"]) + math.log(residual))
