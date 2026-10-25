"""Marginalize unobserved action presence when scoring partial QA observations."""

from __future__ import annotations

import math
from collections import Counter
from functools import cache

from cuhkx.metadata import option_atoms
from cuhkx.submission import available_letters


def canonical_clauses(clauses):
    unique = sorted({tuple(sorted(clause)) for clause in clauses}, key=lambda clause: (len(clause), clause))
    result = []
    for clause in unique:
        if not any(set(previous) <= set(clause) for previous in result):
            result.append(clause)
    return tuple(result)


def negative_clause_probability(clauses, probabilities):
    """Probability that each clause contains at least one absent action."""
    @cache
    def solve(state):
        if not state:
            return 1.
        if not state[0]:
            return 0.
        counts = Counter(atom for clause in state for atom in clause)
        atom = min(counts, key=lambda value: (-counts[value], value))
        absent = canonical_clauses(clause for clause in state if atom not in clause)
        present = canonical_clauses(tuple(value for value in clause if value != atom) for clause in state)
        probability = probabilities[atom]
        return (1 - probability) * solve(absent) + probability * solve(present)

    return solve(canonical_clauses(clauses))


def state_log_likelihood(group, state, action_probabilities):
    """Score an answer tuple without treating unspecified actions as absent."""
    probabilities = {atom: min(max(float(value), 1e-5), 1 - 1e-5)
                     for atom, value in action_probabilities.items()}
    positive, negative, clauses = set(state.positives), set(), []
    answers = state.answer_map()
    for _, row in group.iterrows():
        category = row.category
        if category not in answers:
            continue
        selected = set(answers[category])
        for letter in available_letters(row):
            if letter in selected:
                continue
            atoms = set(option_atoms(row[letter]))
            if category in {"single", "multi"}:
                if len(atoms) != 1:
                    raise ValueError("single/multi action options must be atomic")
                negative.update(atoms)
            elif category == "combination":
                clauses.append(atoms)
    if positive & negative:
        return -math.inf
    unknown_clauses = [clause - positive for clause in clauses if not clause & negative]
    residual = negative_clause_probability(unknown_clauses, probabilities)
    if residual <= 0:
        return -math.inf
    return (sum(math.log(probabilities[atom]) for atom in positive)
            + sum(math.log1p(-probabilities[atom]) for atom in negative) + math.log(residual))
