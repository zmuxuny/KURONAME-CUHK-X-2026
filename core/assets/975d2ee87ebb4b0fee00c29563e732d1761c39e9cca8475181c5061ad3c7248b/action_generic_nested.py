"""Generic video action-set supervision with subject-scoped option evidence."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from cuhkx.actions import HAU_CLASS_TEXT, HAU_TEXT_CLASS
from cuhkx.clip_constraints import repeated_recording_groups
from cuhkx.joint_actions import JOINT_CATEGORIES, action_universe, enumerate_joint_states
from cuhkx.metadata import clip_key, feature_unit, option_atoms, subject_id
from cuhkx.option_frequency import build_option_frequency_features
from cuhkx.sequence_skeleton_nested import array_digest

METADATA = ["qa_id", "source", "path", "category", "A", "B", "C", "D"]


@dataclass
class Prepared:
    frame: pd.DataFrame
    option_matrix: np.ndarray
    option_metadata: pd.DataFrame
    clips: list[dict]


def read_visual(path):
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in ("X", "units", "feature_names")}
    data["units"] = data["units"].astype(str)
    data["feature_names"] = data["feature_names"].astype(str)
    if (data["X"].shape != (len(data["units"]), len(data["feature_names"]))
            or not np.isfinite(data["X"]).all()
            or len(set(data["units"])) != len(data["units"])):
        raise ValueError("visual archive must contain finite unique whole-video descriptors")
    data["index"] = {unit: i for i, unit in enumerate(data["units"])}
    return data


def prepare(qa, *, training):
    # Remove target columns before any option/scene feature utility is called.
    metadata = qa.loc[qa.source.eq("HAU"), METADATA].copy()
    if metadata.qa_id.duplicated().any() or metadata.empty:
        raise ValueError("HAU metadata must be nonempty with unique question IDs")
    repeats = repeated_recording_groups(metadata)
    matrix, option_metadata = build_option_frequency_features(metadata)
    if "label" in option_metadata or "answer" in option_metadata:
        raise ValueError("answer column reached the unlabeled option feature builder")
    options = {(row.qa_id, row.letter): i for i, row in enumerate(option_metadata.itertuples())}
    metadata["clip"] = metadata.path.map(clip_key)
    metadata["subject"] = metadata.path.map(subject_id)
    metadata["unit"] = metadata.path.map(lambda value: feature_unit(value, training=training))
    frame = metadata.loc[metadata.category.isin(JOINT_CATEGORIES)].reset_index(drop=True)
    if training and frame.subject.isna().any():
        raise ValueError("training subject must be parsed from the official media path")
    clips = []
    for clip_id, group in metadata.groupby("clip", sort=False):
        targets = group.loc[group.category.isin(JOINT_CATEGORIES)]
        if targets.empty:
            continue
        if targets.category.duplicated().any() or group.subject.fillna("test").nunique() != 1:
            raise ValueError("joint clip metadata duplicates categories or crosses subjects")
        states = sorted(enumerate_joint_states(group), key=lambda state: state.answers)
        if not states:
            raise ValueError("official options admit no consistent action state")
        option_keys = [(row.qa_id, letter) for row in targets.itertuples() for letter in "ABCD"]
        category_by_id = targets.set_index("qa_id").category.to_dict()
        action_masks, option_masks, answers = [], [], []
        for state in states:
            mask = np.zeros(40, dtype=np.float64)
            for atom in state.positives:
                if atom not in HAU_TEXT_CLASS:
                    raise ValueError("joint action outside the official forty-class vocabulary")
                mask[HAU_TEXT_CLASS[atom]] = 1
            answer_map = state.answer_map()
            action_masks.append(mask)
            option_masks.append([int(letter in answer_map[category_by_id[qa_id]]) for qa_id, letter in option_keys])
            answers.append([answer_map[row.category] for row in targets.itertuples()])
        clips.append({"clip": clip_id, "unit": group.iloc[0].unit, "subject": group.iloc[0].subject,
                      "repeat_group": repeats[clip_id], "qa_ids": targets.qa_id.tolist(),
                      "action_masks": np.asarray(action_masks), "option_masks": np.asarray(option_masks),
                      "answers": answers, "option_rows": np.asarray([options[key] for key in option_keys]),
                      "atom_count": max(1, len(action_universe(group))),
                      "sequence_positives": sorted({atom for row in group.loc[group.category.eq("sequence")].itertuples()
                                                    for letter in "ABCD" for atom in option_atoms(getattr(row, letter))})})
    for group_id in {clip["repeat_group"] for clip in clips}:
        members = [clip for clip in clips if clip["repeat_group"] == group_id]
        if len(members) > 3 or len({"test" if pd.isna(clip["subject"]) else clip["subject"] for clip in members}) != 1:
            raise ValueError("repeated scene crosses subjects or contains more than three videos")
    return Prepared(frame, matrix, option_metadata, clips)


def fitting_arrays(qa, prepared, visual, train, valid):
    train, valid = set(train), set(valid)
    universe = set(prepared.frame.subject)
    if not train or train & valid or (train | valid) - universe:
        raise ValueError("invalid or overlapping fitted subject scope")
    frame = prepared.frame
    videos = frame.loc[frame.subject.isin(train)].drop_duplicates("unit").copy()
    videos = videos.loc[videos.unit.isin(visual["index"])]
    if set(videos.subject) != train:
        raise ValueError("partially labeled video training must cover every fitted subject")
    truth = qa.loc[qa.qa_id.isin(frame.loc[frame.subject.isin(train)].qa_id), ["qa_id", "answer"]].set_index("qa_id").answer
    # -1 means unknown. A correct combination certifies its members, but is not
    # an exhaustive set; an incorrect combination does not negate every member.
    targets = np.full((len(videos), len(HAU_CLASS_TEXT)), -1, dtype=np.int8)
    clips = {clip["unit"]: clip for clip in prepared.clips}
    for index, video in enumerate(videos.itertuples()):
        positives, negatives = set(clips[video.unit]["sequence_positives"]), set()
        for row in frame.loc[frame.unit.eq(video.unit)].itertuples():
            answer = str(truth[row.qa_id])
            if not answer or set(answer) - set("ABCD") or (row.category != "multi" and len(answer) != 1):
                raise ValueError("partial action fitting label is not a valid option answer")
            for letter in answer:
                positives.update(option_atoms(getattr(row, letter)))
            if row.category in {"single", "multi"}:
                for letter in set("ABCD") - set(answer):
                    negatives.update(option_atoms(getattr(row, letter)))
        if positives & negatives:
            raise ValueError("same-video certified action labels contradict each other")
        for value, atoms in [(1, positives), (0, negatives)]:
            for atom in atoms:
                targets[index, HAU_TEXT_CLASS[atom]] = value
    visual_rows = np.asarray([visual["index"][unit] for unit in videos.unit])
    option_rows = np.flatnonzero(prepared.option_metadata.subject.isin(train))
    options = prepared.option_metadata.iloc[option_rows]
    labels = np.asarray([int(row.letter in truth[row.qa_id]) for row in options.itertuples()], dtype=int)
    record = {"train_subjects": sorted(train), "validation_subjects": sorted(valid),
              "training_qa_ids": frame.loc[frame.subject.isin(train)].qa_id.tolist(),
              "prediction_qa_ids": frame.loc[frame.subject.isin(valid)].qa_id.tolist(),
              "action_training_units": videos.unit.tolist(),
              "visual_rows": visual_rows.tolist(), "option_rows": option_rows.tolist(),
              "visual_matrix_and_targets_sha256": array_digest(visual["X"][visual_rows], targets),
              "option_matrix_and_targets_sha256": array_digest(prepared.option_matrix[option_rows], labels),
              "visual_target_shape": list(targets.shape), "option_fitting_rows": len(labels),
              "positive_counts": (targets == 1).sum(axis=0).tolist(),
              "negative_counts": (targets == 0).sum(axis=0).tolist(),
              "unknown_counts": (targets == -1).sum(axis=0).tolist(),
              "known_action_classes": np.flatnonzero((targets == 1).any(axis=0) & (targets == 0).any(axis=0)).tolist()}
    return (visual["X"][visual_rows], targets, prepared.option_matrix[option_rows], labels), record


class PartialActionClassifier:
    """Forty linear probes, each fitted only on that action's known labels."""

    def __init__(self, config, seed):
        self.config = config
        self.seed = seed

    def get_params(self):
        return {**self.config, "seed": self.seed}

    def fit(self, matrix, targets):
        self.scaler = StandardScaler().fit(matrix)
        transformed = self.scaler.transform(matrix)
        self.n_features_in_ = matrix.shape[1]
        self.n_outputs_ = 40
        self.models = []
        self.fitting_rows = []
        parameters = {key: self.config[key] for key in ("C", "class_weight", "solver", "max_iter", "tol")}
        for action in range(40):
            rows = np.flatnonzero(targets[:, action] != -1)
            self.fitting_rows.append(rows)
            estimator = None
            if len(np.unique(targets[rows, action])) == 2:
                estimator = LogisticRegression(**parameters, random_state=self.seed + action)
                estimator.fit(transformed[rows], targets[rows, action])
                if estimator.n_iter_.max() >= parameters["max_iter"]:
                    raise ValueError("partial action classifier did not converge")
            self.models.append(estimator)
        return self

    def predict(self, matrix):
        transformed = self.scaler.transform(matrix)
        probabilities = np.full((len(matrix), 40), 0.5)
        for action, estimator in enumerate(self.models):
            if estimator is not None:
                probabilities[:, action] = estimator.predict_proba(transformed)[:, 1]
        return probabilities


def fit_bundle(qa, prepared, visual, train, valid, config, seed):
    (x_visual, y_visual, x_option, y_option), scope = fitting_arrays(qa, prepared, visual, train, valid)
    presence = PartialActionClassifier(config["presence"], seed)
    frequency = ExtraTreesClassifier(n_estimators=200, min_samples_leaf=5, max_depth=12,
                                     max_features=0.30, class_weight="balanced", n_jobs=3,
                                     random_state=seed + 1000000)
    presence.fit(x_visual, y_visual)
    frequency.fit(x_option, y_option)
    scope["seed"] = seed
    scope["model_parameters"] = {"presence": presence.get_params(), "frequency": frequency.get_params()}
    return {"presence": presence, "frequency": frequency, "training_subjects": sorted(train),
            "known_action_classes": scope["known_action_classes"], "feature_names": visual["feature_names"],
            "config": config}, scope


def evidence(bundle, prepared, visual, qa_ids):
    qa_ids = list(qa_ids)
    selected = set(qa_ids)
    frame = prepared.frame.loc[prepared.frame.qa_id.isin(selected)]
    if len(selected) != len(qa_ids) or set(frame.qa_id) != selected:
        raise ValueError("target QA IDs are not an exact unique action scope")
    if set(frame.subject.dropna()) & set(bundle["training_subjects"]):
        raise ValueError("target subject was used by an upstream action model")
    if not np.array_equal(visual["feature_names"], bundle["feature_names"]):
        raise ValueError("visual feature schema differs from the saved model")
    clips = [clip for clip in prepared.clips if set(clip["qa_ids"]) & selected]
    if any(not set(clip["qa_ids"]) <= selected for clip in clips):
        raise ValueError("target scope splits questions from the same video")
    matrix = np.full((len(clips), 40), 0.5)
    available = np.asarray([clip["unit"] in visual["index"] for clip in clips])
    if available.any():
        rows = [visual["index"][clip["unit"]] for clip in clips if clip["unit"] in visual["index"]]
        matrix[available] = bundle["presence"].predict(visual["X"][rows])
    unknown = sorted(set(range(40)) - set(bundle["known_action_classes"]))
    matrix[:, unknown] = 0.5
    matrix = np.clip(matrix, 1e-5, 1 - 1e-5)
    log_odds = np.log(matrix / (1 - matrix))
    pooled = log_odds.copy()
    pooled_available = available.copy()
    for group_id in {clip["repeat_group"] for clip in clips}:
        rows = np.asarray([i for i, clip in enumerate(clips) if clip["repeat_group"] == group_id])
        present = rows[available[rows]]
        if len(present):
            pooled[rows] = log_odds[present].mean(axis=0)
            pooled_available[rows] = True
    option_rows = np.concatenate([clip["option_rows"] for clip in clips])
    classifier = bundle["frequency"]
    columns = {int(label): i for i, label in enumerate(classifier.classes_)}
    if set(columns) != {0, 1}:
        raise ValueError("option classifier must retain both binary labels")
    option_probabilities = classifier.predict_proba(prepared.option_matrix[option_rows])[:, columns[1]]
    option_probabilities = np.clip(option_probabilities, 1e-5, 1 - 1e-5)
    option_logits = np.log(option_probabilities / (1 - option_probabilities))
    records, start = [], 0
    for i, clip in enumerate(clips):
        stop = start + len(clip["option_rows"])
        records.append({"clip": clip, "action_scores": clip["action_masks"] @ pooled[i],
                        "option_scores": clip["option_masks"] @ option_logits[start:stop],
                        "visual_available": bool(pooled_available[i])})
        start = stop
    return {"qa_ids": qa_ids, "records": records}


def predict(prepared, values, decoder):
    output = {}
    subjects = prepared.frame.set_index("qa_id").subject
    categories = prepared.frame.set_index("qa_id").category
    for item in values["records"]:
        clip = item["clip"]
        # Forest parallel reductions can differ at roundoff scale. Freeze a
        # 12-decimal score tie rule so checkpoint replay cannot change answers.
        scores = np.round(decoder["visual_weight"] * item["action_scores"] + decoder["option_weight"] * item["option_scores"], 12)
        order = np.argsort(-scores, kind="stable")
        margin = float((scores[order[0]] - scores[order[1]]) / clip["atom_count"]) if len(order) > 1 else 0.0
        for qa_id, answer in zip(clip["qa_ids"], clip["answers"][order[0]], strict=True):
            output[qa_id] = {"qa_id": qa_id, "prediction": answer, "margin": margin,
                             "subject": subjects[qa_id], "category": categories[qa_id],
                             "selected_decoder": decoder["name"], "evidence_available": item["visual_available"],
                             "candidate_count": len(order)}
    return pd.DataFrame([output[qa_id] for qa_id in values["qa_ids"]])
