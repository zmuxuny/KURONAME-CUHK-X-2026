"""Clip-level constrained decoding for the three HAU action question families.

The HAU single, multi, and combination questions are different observations of
one latent set of actions.  Sequence options are certified members of that set.
This module enumerates answer tuples that are mutually consistent and ranks the
resulting action sets with externally supplied action-presence probabilities.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

from cuhkx.metadata import option_atoms
from cuhkx.submission import available_letters

JOINT_CATEGORIES = ("single", "multi", "combination")
ACTION_CATEGORIES = frozenset((*JOINT_CATEGORIES, "sequence"))


@dataclass(frozen=True)
class JointActionState:
    """One mutually consistent answer tuple and its implied positive actions."""

    answers: tuple[tuple[str, str], ...]
    positives: frozenset[str]

    def answer_map(self) -> dict[str, str]:
        return dict(self.answers)


@dataclass(frozen=True)
class JointActionDecision:
    """Best joint state plus an atom-normalized runner-up margin."""

    state: JointActionState
    margin: float
    candidate_count: int
    atom_count: int


def _category_rows(group: pd.DataFrame) -> dict[str, pd.Series]:
    rows: dict[str, pd.Series] = {}
    for category, category_group in group.groupby("category", sort=False):
        name = str(category)
        if name not in ACTION_CATEGORIES:
            continue
        if len(category_group) != 1:
            raise ValueError(f"expected one {name!r} row per clip, found {len(category_group)}")
        rows[name] = category_group.iloc[0]
    return rows


def _answer_choices(category: str, row: pd.Series) -> tuple[str, ...]:
    letters = tuple(available_letters(row))
    if category != "multi":
        return letters
    return tuple(
        "".join(subset)
        for size in range(1, len(letters) + 1)
        for subset in itertools.combinations(letters, size)
    )


def _answer_positives(category: str, row: pd.Series, answer: str = "") -> frozenset[str]:
    if category == "sequence":
        selected = available_letters(row)
    elif category == "combination":
        selected = (answer,)
    else:
        selected = tuple(answer)
    return frozenset(
        atom
        for letter in selected
        for atom in option_atoms(row[letter])
    )


def action_universe(group: pd.DataFrame) -> frozenset[str]:
    """Return every action atom mentioned by an action question in one clip."""

    atoms: set[str] = set()
    for _, row in group.loc[group["category"].isin(ACTION_CATEGORIES)].iterrows():
        for letter in available_letters(row):
            atoms.update(option_atoms(row[letter]))
    return frozenset(atoms)


def enumerate_joint_states(group: pd.DataFrame) -> tuple[JointActionState, ...]:
    """Enumerate all answer tuples whose implied action set is self-consistent.

    A single option is correct exactly when its action is in the latent set.  A
    multi answer contains every offered action in the set.  A combination
    option is correct exactly when all of its action atoms are in the set.
    Sequence options are all known positives, but their order is handled by a
    separate temporal decoder.
    """

    rows = _category_rows(group)
    present_categories = tuple(category for category in JOINT_CATEGORIES if category in rows)
    if not present_categories:
        return ()

    fixed = (
        _answer_positives("sequence", rows["sequence"])
        if "sequence" in rows
        else frozenset()
    )
    result: list[JointActionState] = []
    choice_product = itertools.product(
        *(_answer_choices(category, rows[category]) for category in present_categories)
    )
    for values in choice_product:
        answers = dict(zip(present_categories, values, strict=True))
        positives = fixed.union(
            *(
                _answer_positives(category, rows[category], answers[category])
                for category in present_categories
            )
        )
        consistent = True
        for category in present_categories:
            row = rows[category]
            contained = [
                letter
                for letter in available_letters(row)
                if set(option_atoms(row[letter])) <= positives
            ]
            if category == "multi":
                implied = "".join(contained)
            else:
                implied = contained[0] if len(contained) == 1 else ""
            if implied != answers[category]:
                consistent = False
                break
        if consistent:
            result.append(
                JointActionState(
                    answers=tuple((category, answers[category]) for category in present_categories),
                    positives=frozenset(positives),
                )
            )
    return tuple(result)


def decode_joint_state(
    group: pd.DataFrame,
    action_probabilities: Mapping[str, float],
    *,
    epsilon: float = 1e-5,
) -> JointActionDecision:
    """Rank consistent states under independent Bernoulli action evidence.

    Terms for absent actions are identical up to a clip-level constant, so the
    score is the sum of action log-odds over the state's positive atoms.  The
    returned margin is divided by the number of mentioned atoms to make it
    comparable across clips with differently sized option sets.
    """

    if not 0 < epsilon < 0.5:
        raise ValueError("epsilon must be in (0, 0.5)")
    states = enumerate_joint_states(group)
    if not states:
        raise ValueError("clip has no consistent joint action state")
    universe = action_universe(group)
    missing = sorted(universe - set(action_probabilities))
    if missing:
        raise ValueError(f"missing action probabilities: {missing}")

    log_odds = {
        atom: math.log(probability / (1.0 - probability))
        for atom in universe
        for probability in [min(max(float(action_probabilities[atom]), epsilon), 1.0 - epsilon)]
    }
    ranked = sorted(
        (
            (sum(log_odds[atom] for atom in state.positives), state)
            for state in states
        ),
        key=lambda item: (-item[0], item[1].answers),
    )
    atom_count = max(1, len(universe))
    margin = (
        (ranked[0][0] - ranked[1][0]) / atom_count
        if len(ranked) > 1
        else math.inf
    )
    return JointActionDecision(
        state=ranked[0][1],
        margin=float(margin),
        candidate_count=len(states),
        atom_count=len(universe),
    )


def decode_joint_state_with_option_evidence(
    group: pd.DataFrame,
    action_probabilities: Mapping[str, float],
    option_probabilities: Mapping[str, Mapping[str, float]],
    *,
    option_weight: float,
    epsilon: float = 1e-5,
) -> JointActionDecision:
    """Rank states with action evidence plus per-question option likelihoods.

    ``option_probabilities`` supplies an independent Bernoulli probability for
    every valid option letter in each present joint-category question.  This is
    useful for evidence such as repeated-recording option frequencies that is
    naturally expressed in answer space instead of the global action space.
    A zero weight deliberately delegates to :func:`decode_joint_state`, making
    backward-compatible reproduction an executable invariant.
    """

    if option_weight < 0:
        raise ValueError("option_weight must be nonnegative")
    if option_weight == 0:
        return decode_joint_state(group, action_probabilities, epsilon=epsilon)
    if not 0 < epsilon < 0.5:
        raise ValueError("epsilon must be in (0, 0.5)")

    states = enumerate_joint_states(group)
    if not states:
        raise ValueError("clip has no consistent joint action state")
    universe = action_universe(group)
    missing = sorted(universe - set(action_probabilities))
    if missing:
        raise ValueError(f"missing action probabilities: {missing}")
    action_log_odds = {
        atom: math.log(probability / (1.0 - probability))
        for atom in universe
        for probability in [
            min(max(float(action_probabilities[atom]), epsilon), 1.0 - epsilon)
        ]
    }
    rows = _category_rows(group)
    joint_rows = {
        category: row
        for category, row in rows.items()
        if category in JOINT_CATEGORIES
    }

    ranked: list[tuple[float, JointActionState]] = []
    for state in states:
        answers = state.answer_map()
        option_log_likelihood = 0.0
        for category, row in joint_rows.items():
            qa_id = str(row["qa_id"])
            if qa_id not in option_probabilities:
                raise ValueError(f"option probabilities lack {qa_id!r}")
            answer = answers[category]
            for letter in available_letters(row):
                if letter not in option_probabilities[qa_id]:
                    raise ValueError(
                        f"option probabilities for {qa_id!r} lack {letter!r}"
                    )
                probability = min(
                    max(float(option_probabilities[qa_id][letter]), epsilon),
                    1.0 - epsilon,
                )
                positive = letter in answer if category == "multi" else letter == answer
                option_log_likelihood += math.log(
                    probability if positive else 1.0 - probability
                )
        action_score = sum(action_log_odds[atom] for atom in state.positives)
        ranked.append((action_score + option_weight * option_log_likelihood, state))

    ranked.sort(key=lambda item: (-item[0], item[1].answers))
    atom_count = max(1, len(universe))
    margin = (
        (ranked[0][0] - ranked[1][0]) / atom_count
        if len(ranked) > 1
        else math.inf
    )
    return JointActionDecision(
        state=ranked[0][1],
        margin=float(margin),
        candidate_count=len(states),
        atom_count=len(universe),
    )
