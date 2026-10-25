"""Deployment-context interventions on frozen subject-excluded emotion models."""

import numpy as np
import pandas as pd

from cuhkx import emotion_recipe_nested as recipe
from cuhkx.emotion_nested import prepare

TARGET_COLUMNS = ("answer", "truth_word", "prediction", "target", "label", "mask", "correct")
CONDITIONS = ("original_control", "without_action_question_context", "independent_recordings")


def make_views(questions):
    """Only the original preparation reads Train labels for the later metric join."""
    original, original_context = prepare(questions, training=True)
    scenes = original[["clip", "group", "position"]].copy()
    emotion_only = questions.loc[questions.source.eq("HAU") & questions.category.eq("emotion")]
    no_context, contexts = prepare(emotion_only, training=True, manifest=scenes)
    # A singleton has no available relative recording index; zero is a placeholder,
    # not an inferred true manner label. Preserve this choice in the experiment spec.
    single_scenes = scenes.assign(group=scenes["clip"].map(lambda value: "singleton:" + value), position=0)
    single, single_context = prepare(questions, training=True, manifest=single_scenes)
    views = {"original_control": (original, original_context),
             "without_action_question_context": (no_context, contexts),
             "independent_recordings": (single, single_context)}
    for frame, _ in views.values():
        if frame.qa_id.tolist() != original.qa_id.tolist():
            raise ValueError("intervention changed the official emotion row axis")
        if not frame[["unit", "subject", "options", "letters"]].equals(
            original[["unit", "subject", "options", "letters"]]
        ):
            raise ValueError("intervention changed the target media, subjects or options")
    return views


def inference_frame(frame, *, poison=False):
    clean = frame.drop(columns=list(TARGET_COLUMNS), errors="ignore").copy()
    if poison:
        for name in TARGET_COLUMNS:
            clean[name] = "DO_NOT_READ_THIS_TARGET"
    return clean


def score_subject(bundle, frame, contexts, matrices, masks, subject, config, policy, *, poison=False):
    if subject in bundle["training_subjects"]:
        raise ValueError("held-out subject appears in upstream training")
    safe = inference_frame(frame, poison=poison)
    rows = np.flatnonzero(safe.subject.eq(subject))
    if not len(rows):
        raise ValueError("missing held-out subject rows")
    evidence = recipe.recipe_evidence(bundle, safe, contexts, matrices, masks, rows, config)
    output = recipe.predict_rows(safe.iloc[rows], evidence, policy)
    if output.qa_id.tolist() != safe.iloc[rows].qa_id.tolist():
        raise ValueError("decoder changed held-out IDs or order")
    return output.assign(subject=subject)


def control_matches(expected, output):
    if not expected.qa_id.is_unique or set(expected.qa_id) != set(output.qa_id):
        raise ValueError("control coverage differs")
    aligned = expected.set_index("qa_id").loc[output.qa_id].reset_index()
    if aligned.prediction.tolist() != output.prediction.tolist():
        raise ValueError("original control does not reproduce its frozen parent")
    if "margin" in aligned:
        np.testing.assert_allclose(aligned.margin.astype(float), output.margin.astype(float), rtol=1e-12, atol=1e-12)
    return len(output)


def concatenate(outputs, ids):
    result = pd.concat(outputs, ignore_index=True)
    if not result.qa_id.is_unique or set(result.qa_id) != set(ids):
        raise ValueError("outer predictions do not cover the full official scope")
    return result.set_index("qa_id").loc[ids].reset_index()
