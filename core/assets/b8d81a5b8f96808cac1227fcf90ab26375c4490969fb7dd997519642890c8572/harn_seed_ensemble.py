"""Fixed equal averaging of two equally scoped, subject-adapted HARn models."""

from __future__ import annotations

import numpy as np

from cuhkx.harn import action_class, action_text_classes, object_text_classes
from cuhkx.harn_nested import CLASSES, OFFICIAL_SUBJECTS, fit_prior
from cuhkx.harn_tail import KIND as TAIL_KIND
from cuhkx.metadata import subject_id
from cuhkx.video_midfeatures import SPEC

NAMES = ("original", "repeated")
SEEDS = {"original": 20262905, "repeated": 20262906}
KEYS = {"units", "skeleton_probabilities", "skeleton_present", "skeleton_classes",
        "visual_probabilities", "visual_present", "visual_classes"}


def mappings(qa):
    return {"action_mapping": action_text_classes(qa),
            "object_mapping": object_text_classes(qa), "prior": fit_prior(qa)}


def validate_evidence(evidence):
    if set(evidence) != KEYS:
        raise ValueError("unexpected retained probability schema")
    units = evidence["units"]
    if units.ndim != 1 or len(set(units)) != len(units):
        raise ValueError("probability units must be unique")
    for family in ("skeleton", "visual"):
        classes = evidence[f"{family}_classes"]
        present = evidence[f"{family}_present"]
        values = evidence[f"{family}_probabilities"]
        if (classes.ndim != 1 or not len(classes) or not np.issubdtype(classes.dtype, np.integer)
                or not np.array_equal(classes, np.unique(classes)) or not set(classes) <= set(CLASSES)):
            raise ValueError("probability class axis differs from official seen-class indices")
        if present.shape != (len(units),) or present.dtype != np.bool_:
            raise ValueError("missing-modality mask must be a boolean unit axis")
        if values.shape != (len(units), 44) or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("invalid aligned class probabilities")
        if np.any(values[~present]) or np.any(values[:, np.setdiff1d(CLASSES, classes)]):
            raise ValueError("missing modalities and unseen classes must have zero probability")
        np.testing.assert_allclose(values[present].sum(axis=1), 1, rtol=0, atol=1e-6)


def equal_evidence(original, repeated):
    """There is deliberately no argument selecting the visual mixing weight."""
    for evidence in (original, repeated):
        validate_evidence(evidence)
    for key in KEYS - {"visual_probabilities"}:
        if key == "skeleton_probabilities":
            np.testing.assert_allclose(original[key], repeated[key], rtol=0, atol=1e-12)
        else:
            np.testing.assert_array_equal(original[key], repeated[key])
    result = {key: value.copy() for key, value in original.items()}
    result["visual_probabilities"] = .5 * original["visual_probabilities"] + .5 * repeated["visual_probabilities"]
    return result


def check_scope(scope, qa, train, valid):
    train, valid = set(train), set(valid)
    if train & valid or train | valid != OFFICIAL_SUBJECTS:
        raise ValueError("scope must exclude every outer and inner validation subject")
    if set(scope["train_subjects"]) != train or set(qa.path.map(subject_id)) != train:
        raise ValueError("actual fitting subjects differ")
    if scope["training_qa_ids"] != qa.qa_id.tolist():
        raise ValueError("actual dictionary fitting QA rows differ")
    if set(scope["dictionary_train_subjects"]) != train:
        raise ValueError("dictionary fitting subjects differ")
    for family in ("skeleton", "visual"):
        feature = scope["features"][family]
        units, labels = feature["units"], feature["labels"]
        if len(set(units)) != len(units) or not units:
            raise ValueError("fitting feature units must be nonempty and unique")
        if any(not unit.startswith("Training/HARn/") or subject_id(unit) not in train for unit in units):
            raise ValueError("non-HARn or held-out feature used for fitting")
        if labels != [action_class(unit) for unit in units] or not set(labels) <= set(CLASSES):
            raise ValueError("actual training action labels differ")
        if set(feature["train_subjects"]) != {subject_id(unit) for unit in units}:
            raise ValueError("feature subject descriptor differs from actual units")
    adapted = scope["adapted_visual"]
    if (adapted["labels"] != scope["features"]["visual"]["labels"]
            or len(adapted["rows"]) != len(adapted["labels"])
            or len(set(adapted["rows"])) != len(adapted["rows"])
            or not adapted["retained_before_validation"] or adapted["middle_spec"] != SPEC):
        raise ValueError("retained adapted fitting rows or feature contract differ")


def check_model(bundle, scope, qa, seed, config):
    if bundle["kind"] != TAIL_KIND or bundle["status"] != "diagnostic_only":
        raise ValueError("expected a retained subject-adapted HARn model")
    if bundle["config"] != config or bundle["train_subjects"] != scope["train_subjects"]:
        raise ValueError("retained model configuration or fitting subjects differ")
    if bundle["visual_spec"] != SPEC:
        raise ValueError("retained visual specification differs")
    for key, value in mappings(qa).items():
        if bundle[key] != value:
            raise ValueError("retained dictionary was not fitted on the declared QA scope")
    learned = bundle["visual_model"]
    if learned.seed != seed or learned.settings != config["visual"]["adaptation"]:
        raise ValueError("retained visual seed or adaptation settings differ")
    adapted = scope["adapted_visual"]
    np.testing.assert_array_equal(learned.training_rows, adapted["rows"])
    np.testing.assert_array_equal(learned.training_labels, adapted["labels"])
    np.testing.assert_array_equal(learned.classes_, np.unique(adapted["labels"]))
    np.testing.assert_array_equal(bundle["visual_classes"], learned.classes_)
    np.testing.assert_array_equal(bundle["skeleton_classes"], np.unique(scope["features"]["skeleton"]["labels"]))
    if learned.history != adapted["history"] or learned.state["3.weight"].shape != (44, 512):
        raise ValueError("retained training history or44-class head differs")
    if any(np.issubdtype(value.dtype, np.floating) and value.dtype != np.float16
           for value in learned.state.values()):
        raise ValueError("retained visual floating state must use pre-validation fp16")


def check_final_models(models):
    for name, bundle in models.items():
        if (bundle["kind"] != TAIL_KIND or set(bundle["train_subjects"]) != OFFICIAL_SUBJECTS
                or bundle["visual_model"].seed != SEEDS[name] or bundle["visual_spec"] != SPEC):
            raise ValueError("final model kind, subjects, seed or feature contract differ")
    for key in ("action_mapping", "object_mapping", "prior"):
        if models["original"][key] != models["repeated"][key]:
            raise ValueError("final model dictionaries differ")
    for key in ("skeleton_names", "skeleton_classes", "visual_classes"):
        np.testing.assert_array_equal(models["original"][key], models["repeated"][key])
    if models["original"]["visual_model"].settings != models["repeated"]["visual_model"].settings:
        raise ValueError("the two final visual adaptation recipes differ")
