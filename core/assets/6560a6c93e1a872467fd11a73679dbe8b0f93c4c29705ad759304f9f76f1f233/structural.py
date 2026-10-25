"""Media-free structural baseline using only permitted question metadata.

This is inspired by the publicly shared Fususu clip-graph decoder, but is implemented as a
standalone estimator with strict option masking and a subject-disjoint evaluation path.
"""

from __future__ import annotations

import itertools
import math
from collections import Counter
from dataclasses import dataclass

import pandas as pd

from cuhkx.metadata import clip_key, option_atoms
from cuhkx.submission import LETTERS


@dataclass(frozen=True)
class StructuralConfig:
    prior_pseudopositive: float = 2.0
    prior_pseudototal: float = 8.0
    prior_weight: float = 0.25
    multi_pattern_smoothing: float = 4.0
    multi_atom_blend: float = 0.10
    multi_cardinality_weight: float = 2.0
    sequence_smoothing: float = 1.5


class StructuralDecoder:
    def __init__(self, config: StructuralConfig | None = None):
        self.config = config or StructuralConfig()
        self.atom_seen: Counter[tuple[str, str, str]] = Counter()
        self.atom_correct: Counter[tuple[str, str, str]] = Counter()
        self.multi_pattern_seen: Counter[tuple[int, int, int]] = Counter()
        self.multi_pattern_correct: Counter[tuple[int, int, int]] = Counter()
        self.multi_sizes: Counter[int] = Counter()
        self.before: Counter[tuple[str, str]] = Counter()
        self._fitted = False

    @staticmethod
    def _valid_options(row: pd.Series) -> dict[str, tuple[str, ...]]:
        return {
            letter: option_atoms(row.get(letter, ""))
            for letter in LETTERS
            if str(row.get(letter, "")).strip()
        }

    @staticmethod
    def _clip_counts(group: pd.DataFrame) -> dict[str, Counter[str]]:
        result: dict[str, Counter[str]] = {}
        for category, rows in group.groupby("category", sort=False):
            count: Counter[str] = Counter()
            for _, row in rows.iterrows():
                for atoms in StructuralDecoder._valid_options(row).values():
                    count.update(atoms)
            result[str(category)] = count
        return result

    @staticmethod
    def _multi_pattern(atom: str, counts: dict[str, Counter[str]]) -> tuple[int, int, int]:
        return (
            counts.get("single", Counter())[atom],
            counts.get("combination", Counter())[atom],
            counts.get("sequence", Counter())[atom],
        )

    def fit(self, frame: pd.DataFrame) -> StructuralDecoder:
        required = {"qa_id", "source", "path", "category", "answer", *LETTERS}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"training frame missing columns: {missing}")

        # Refit replaces the learned state, including when a model is reused across folds.
        self._fitted = False
        for counts in (
            self.atom_seen, self.atom_correct, self.multi_pattern_seen,
            self.multi_pattern_correct, self.multi_sizes, self.before,
        ):
            counts.clear()

        for _, row in frame.iterrows():
            answer = str(row["answer"]).strip().upper()
            category = str(row["category"])
            source = str(row["source"])
            options = self._valid_options(row)
            if category == "sequence":
                ordered = [options[letter][0] for letter in answer]
                for left_index, left in enumerate(ordered):
                    for right in ordered[left_index + 1 :]:
                        self.before[(left, right)] += 1
                continue
            for letter, atoms in options.items():
                for atom in atoms:
                    key = (source, category, atom)
                    self.atom_seen[key] += 1
                    self.atom_correct[key] += int(letter in answer)

        hau = frame[frame["source"] == "HAU"].copy()
        hau["_clip_key"] = hau["path"].map(clip_key)
        for _, group in hau.groupby("_clip_key", sort=False):
            counts = self._clip_counts(group)
            for _, row in group[group["category"] == "multi"].iterrows():
                answer = str(row["answer"]).strip().upper()
                self.multi_sizes[len(answer)] += 1
                for letter, atoms in self._valid_options(row).items():
                    # Multi options contain one action in the live data; max keeps this robust.
                    patterns = [self._multi_pattern(atom, counts) for atom in atoms]
                    pattern = max(patterns) if patterns else (0, 0, 0)
                    self.multi_pattern_seen[pattern] += 1
                    self.multi_pattern_correct[pattern] += int(letter in answer)

        self._fitted = True
        return self

    def _atom_probability(self, source: str, category: str, atoms: tuple[str, ...]) -> float:
        cfg = self.config
        probabilities = []
        for atom in atoms:
            key = (source, category, atom)
            probabilities.append(
                (self.atom_correct[key] + cfg.prior_pseudopositive)
                / (self.atom_seen[key] + cfg.prior_pseudototal)
            )
        return sum(probabilities) / len(probabilities)

    def _single_prediction(
        self, row: pd.Series, counts: dict[str, Counter[str]]
    ) -> tuple[str, float]:
        cfg = self.config
        category = str(row["category"])
        source = str(row["source"])
        scores: dict[str, float] = {}
        for letter, atoms in self._valid_options(row).items():
            support = sum(
                sum(counter[atom] for atom in atoms)
                for other_category, counter in counts.items()
                if other_category not in {category, "emotion"}
            )
            probability = min(max(self._atom_probability(source, category, atoms), 1e-6), 1 - 1e-6)
            scores[letter] = support + cfg.prior_weight * math.log(probability / (1 - probability))
        ranked = sorted(scores, key=lambda letter: (-scores[letter], letter))
        margin = scores[ranked[0]] - scores[ranked[1]] if len(ranked) > 1 else math.inf
        return ranked[0], margin

    def _multi_prediction(
        self, row: pd.Series, counts: dict[str, Counter[str]]
    ) -> tuple[str, float]:
        cfg = self.config
        source = str(row["source"])
        category = str(row["category"])
        options = self._valid_options(row)
        base = sum(self.multi_pattern_correct.values()) / max(1, sum(self.multi_pattern_seen.values()))
        probabilities: dict[str, float] = {}
        for letter, atoms in options.items():
            patterns = [self._multi_pattern(atom, counts) for atom in atoms]
            pattern = max(patterns) if patterns else (0, 0, 0)
            pattern_probability = (
                self.multi_pattern_correct[pattern] + cfg.multi_pattern_smoothing * base
            ) / (self.multi_pattern_seen[pattern] + cfg.multi_pattern_smoothing)
            atom_probability = self._atom_probability(source, category, atoms)
            probability = (
                (1 - cfg.multi_atom_blend) * pattern_probability
                + cfg.multi_atom_blend * atom_probability
            )
            probabilities[letter] = min(max(probability, 1e-6), 1 - 1e-6)

        total_sizes = sum(self.multi_sizes.values())
        ranked: list[tuple[float, str]] = []
        valid_letters = list(options)
        for length in range(1, len(valid_letters) + 1):
            for subset in itertools.combinations(valid_letters, length):
                selected = set(subset)
                score = sum(
                    math.log(probabilities[letter] if letter in selected else 1 - probabilities[letter])
                    for letter in valid_letters
                )
                score += cfg.multi_cardinality_weight * math.log(
                    (self.multi_sizes[length] + 1) / (total_sizes + len(valid_letters))
                )
                ranked.append((score, "".join(subset)))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return ranked[0][1], ranked[0][0] - ranked[1][0]

    def _sequence_prediction(self, row: pd.Series) -> tuple[str, float]:
        cfg = self.config
        options = self._valid_options(row)
        ranked: list[tuple[float, str]] = []
        for order in itertools.permutations(options):
            score = 0.0
            for left_index, left_letter in enumerate(order):
                left = options[left_letter][0]
                for right_letter in order[left_index + 1 :]:
                    right = options[right_letter][0]
                    left_first = self.before[(left, right)]
                    right_first = self.before[(right, left)]
                    score += math.log(
                        (left_first + cfg.sequence_smoothing)
                        / (left_first + right_first + 2 * cfg.sequence_smoothing)
                    )
            ranked.append((score, "".join(order)))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return ranked[0][1], ranked[0][0] - ranked[1][0]

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("fit must be called before predict")
        work = frame.copy()
        work["_clip_key"] = work["path"].map(clip_key)
        records: list[dict[str, object]] = []
        for _, group in work.groupby("_clip_key", sort=False):
            counts = self._clip_counts(group)
            for _, row in group.iterrows():
                category = str(row["category"])
                if category == "multi":
                    prediction, margin = self._multi_prediction(row, counts)
                elif category == "sequence":
                    prediction, margin = self._sequence_prediction(row)
                else:
                    prediction, margin = self._single_prediction(row, counts)
                records.append(
                    {"qa_id": row["qa_id"], "prediction": prediction, "margin": margin}
                )
        result = pd.DataFrame(records).set_index("qa_id").reindex(frame["qa_id"]).reset_index()
        return result
