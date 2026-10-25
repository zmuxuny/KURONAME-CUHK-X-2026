"""Same-recording question-subset augmentation with conserved option weights."""
from __future__ import annotations

import numpy as np
import pandas as pd

from cuhkx.action_explicit_layout import prepare
from cuhkx.action_generic_nested import METADATA
from cuhkx.metadata import clip_key
from cuhkx.option_frequency_explicit import build_option_frequency_features

CATEGORIES = ('single', 'multi', 'combination', 'sequence')
CONDITIONS = ('full_same_clip', 'no_sequence', 'single_only', 'multi_only', 'combination_only')


def category_masks(metadata):
    frame = metadata.loc[metadata.source.eq('HAU'), METADATA].copy()
    frame['clip'] = frame.path.map(clip_key)
    masks = frame.groupby('clip').category.agg(lambda values: sum(1 << i for i, name in enumerate(CATEGORIES) if name in set(values)))
    return frame, masks.to_dict()


def bits_for_options(metadata, options):
    frame, masks = category_masks(metadata)
    qa_clip = frame.set_index('qa_id')['clip'].to_dict()
    return np.asarray([[bool(masks[qa_clip[q]] & (1 << i)) for i in range(4)] for q in options.qa_id], dtype=np.float32)


def prepare_visible(metadata):
    result = prepare(metadata, training=True, groups=None)
    result.option_matrix = np.column_stack((result.option_matrix, bits_for_options(metadata, result.option_metadata)))
    return result


def augmented_features(metadata):
    """Each actual clip subset appears once; every retained target keeps its options."""
    frame, masks = category_masks(metadata)
    matrices, records = [], []
    for mask in range(1, 16):
        if not mask & 7:
            continue
        clips = {clip for clip, available in masks.items() if available & mask == mask}
        categories = {name for i, name in enumerate(CATEGORIES) if mask & (1 << i)}
        view = frame.loc[frame['clip'].isin(clips) & frame.category.isin(categories), METADATA]
        if view.empty:
            continue
        matrix, options = build_option_frequency_features(view, groups={clip: clip for clip in clips})
        options['visible_mask'] = mask
        options['clip'] = options.qa_id.map(frame.set_index('qa_id')['clip'])
        options['nviews'] = options['clip'].map(lambda clip: 2 ** (masks[clip].bit_count() - 1))
        matrices.append(np.column_stack((matrix, bits_for_options(view, options))))
        records.append(options)
    records = pd.concat(records, ignore_index=True)
    assert not records.duplicated(['qa_id', 'letter', 'visible_mask']).any()
    counts = records.groupby(['qa_id', 'letter']).size()
    expected = records.groupby(['qa_id', 'letter']).nviews.first()
    pd.testing.assert_series_equal(counts, expected, check_names=False)
    return np.concatenate(matrices), records


def weighted_training(matrix, records, truth, subjects):
    chosen = records.subject.isin(subjects).to_numpy()
    rows = records.loc[chosen].reset_index(drop=True)
    labels = np.asarray([int(row.letter in truth[row.qa_id]) for row in rows.itertuples()], dtype=np.int64)
    original = rows.loc[~rows.duplicated(['qa_id', 'letter'])]
    original_labels = np.asarray([int(row.letter in truth[row.qa_id]) for row in original.itertuples()], dtype=int)
    counts = np.bincount(original_labels, minlength=2)
    assert (counts > 0).all()
    balance = len(original_labels) / (2 * counts)
    weights = balance[labels] / rows.nviews.to_numpy()
    audit = rows.assign(label=labels, weight=weights)
    totals = audit.groupby(['qa_id', 'letter']).weight.sum().to_numpy()
    np.testing.assert_allclose(totals, audit.groupby(['qa_id', 'letter']).label.first().map(dict(enumerate(balance))), rtol=0, atol=1e-12)
    return matrix[chosen], labels, weights, audit, balance


def condition_frame(metadata, condition):
    if condition == 'full_same_clip':
        return metadata.copy()
    if condition == 'no_sequence':
        return metadata.loc[~metadata.category.eq('sequence')].copy()
    return metadata.loc[metadata.category.eq(condition.removesuffix('_only'))].copy()
