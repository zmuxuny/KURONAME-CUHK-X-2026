"""Strict submission validation for the CUHK-X Large Model Track."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

LETTERS = "ABCD"
SINGLE_CATEGORIES = {"single", "combination", "emotion", "object_interaction"}
MULTI_CATEGORY = "multi"
SEQUENCE_CATEGORY = "sequence"
KNOWN_CATEGORIES = SINGLE_CATEGORIES | {MULTI_CATEGORY, SEQUENCE_CATEGORY}


class SubmissionValidationError(ValueError):
    """Raised when a candidate violates the live sample-submission contract."""


@dataclass(frozen=True)
class ValidationReport:
    path: str
    rows: int
    columns: tuple[str, ...]
    prediction_column: str
    sha256: str
    size_bytes: int
    valid: bool = True

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def read_csv_str(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def available_letters(row: pd.Series) -> str:
    return "".join(letter for letter in LETTERS if str(row.get(letter, "")).strip())


def validate_prediction(value: str, category: str, valid_letters: str) -> str | None:
    """Return an error description, or ``None`` when a prediction is valid."""

    if value != value.strip():
        return "leading or trailing whitespace"
    value = value.upper()
    if not value:
        return "empty prediction"
    if value != str(value):
        return "prediction is not a string"
    if any(letter not in valid_letters for letter in value):
        return f"uses unavailable letter; valid letters are {valid_letters}"
    if len(set(value)) != len(value):
        return "contains duplicate letters"

    if category in SINGLE_CATEGORIES:
        return None if len(value) == 1 else "single-answer category requires one letter"
    if category == MULTI_CATEGORY:
        if value != "".join(sorted(value)):
            return "multi-answer letters must use canonical alphabetical order"
        return None
    if category == SEQUENCE_CATEGORY:
        if len(value) != len(valid_letters) or set(value) != set(valid_letters):
            return f"sequence must be a permutation of {valid_letters}"
        return None
    return f"unknown category {category!r}"


def validate_submission_frames(
    test: pd.DataFrame,
    sample: pd.DataFrame,
    submission: pd.DataFrame,
) -> str:
    """Validate frames and return the prediction column name."""

    required_test = {"qa_id", "category", *LETTERS}
    missing_test = sorted(required_test - set(test.columns))
    if missing_test:
        raise SubmissionValidationError(f"test CSV missing columns: {missing_test}")

    if len(sample.columns) != 2 or sample.columns[0] != "qa_id":
        raise SubmissionValidationError(
            f"sample must have exactly qa_id plus one prediction column; got {sample.columns.tolist()}"
        )
    prediction_column = str(sample.columns[1])
    expected_columns = ["qa_id", prediction_column]
    if submission.columns.tolist() != expected_columns:
        raise SubmissionValidationError(
            f"columns/order mismatch: expected {expected_columns}, got {submission.columns.tolist()}"
        )

    if len(test) != len(sample) or len(submission) != len(sample):
        raise SubmissionValidationError(
            f"row-count mismatch: test={len(test)}, sample={len(sample)}, submission={len(submission)}"
        )
    if not test["qa_id"].is_unique:
        raise SubmissionValidationError("test qa_id values are not unique")
    if not sample["qa_id"].is_unique:
        raise SubmissionValidationError("sample qa_id values are not unique")
    if not submission["qa_id"].is_unique:
        raise SubmissionValidationError("submission qa_id values are not unique")

    test_ids = test["qa_id"].tolist()
    if sample["qa_id"].tolist() != test_ids:
        raise SubmissionValidationError("sample qa_id order does not exactly match test CSV")
    if submission["qa_id"].tolist() != test_ids:
        raise SubmissionValidationError("submission qa_id order does not exactly match test CSV")

    errors: list[str] = []
    for row_number, (test_row, prediction) in enumerate(
        zip(test.to_dict("records"), submission[prediction_column].tolist(), strict=True),
        start=2,
    ):
        category = str(test_row["category"]).strip()
        valid_letters = "".join(
            letter for letter in LETTERS if str(test_row.get(letter, "")).strip()
        )
        error = validate_prediction(str(prediction), category, valid_letters)
        if error:
            errors.append(f"row {row_number} qa_id={test_row['qa_id']}: {error}")
            if len(errors) == 20:
                break
    if errors:
        raise SubmissionValidationError("invalid predictions:\n" + "\n".join(errors))
    return prediction_column


def validate_submission_files(
    test_path: str | Path,
    sample_path: str | Path,
    submission_path: str | Path,
) -> ValidationReport:
    test_path = Path(test_path)
    sample_path = Path(sample_path)
    submission_path = Path(submission_path).resolve()
    test = read_csv_str(test_path)
    sample = read_csv_str(sample_path)
    submission = read_csv_str(submission_path)
    prediction_column = validate_submission_frames(test, sample, submission)
    payload = submission_path.read_bytes()
    return ValidationReport(
        path=str(submission_path),
        rows=len(submission),
        columns=tuple(submission.columns),
        prediction_column=prediction_column,
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
    )

