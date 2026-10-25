"""Single-question emotion student with no recording identity or peer inputs."""
from __future__ import annotations

import hashlib
import re
from itertools import pairwise

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from cuhkx.metadata import feature_unit, normalize_text

LETTERS = ('A', 'B', 'C', 'D')
MODALITIES = ('imu', 'skeleton', 'radar', 'visual')


def text_vector(text, width):
    text = normalize_text(str(text))
    words = re.findall(r'\w+', text)
    tokens = ['w:' + word for word in words]
    tokens += ['b:' + a + ' ' + b for a, b in pairwise(words)]
    padded = '^' + text + '$'
    tokens += [f'c{size}:' + padded[i:i + size] for size in (3, 4, 5)
               for i in range(max(0, len(padded) - size + 1))]
    vector = np.zeros(width, dtype=np.float32)
    for token in tokens:
        digest = hashlib.sha256(token.encode()).digest()
        index = int.from_bytes(digest[:8], 'little') % width
        vector[index] += 1 if digest[8] & 1 else -1
    return vector / max(float(np.linalg.norm(vector)), 1.)


def local_inputs(questions, banks, *, training):
    """Metadata selects a bank row; only its values and current question enter X.

    This deliberately never calls scene preparation or action-question context
    extraction. IDs, subjects, paths and targets are not returned as features.
    """
    frame = questions.loc[questions.source.eq('HAU') & questions.category.eq('emotion'),
                          ['qa_id', 'path', 'question', *LETTERS]].reset_index(drop=True)
    if not frame.qa_id.is_unique:
        raise ValueError('emotion IDs must be unique for bookkeeping')
    count = len(frame)
    media, masks = {}, np.zeros((count, len(MODALITIES)), dtype=bool)
    for index, name in enumerate(MODALITIES):
        source = banks[name]
        raw = np.zeros((count, len(source['names'])), dtype=np.float32)
        for row, path in enumerate(frame.path):
            unit = feature_unit(path, training=training)
            if unit in source['index']:
                raw[row] = source['matrix'][source['index'][unit]]
                masks[row, index] = True
        if not np.isfinite(raw).all():
            raise ValueError('nonfinite local sensor input')
        media[name] = raw
    options = np.zeros((count, 4, 128), dtype=np.float32)
    visible = np.zeros((count, 4), dtype=bool)
    for i, row in enumerate(frame.itertuples(index=False)):
        for j, letter in enumerate(LETTERS):
            text = str(getattr(row, letter)).strip()
            if text:
                options[i, j] = text_vector(text, 128)
                visible[i, j] = True
    if not visible.any(axis=1).all():
        raise ValueError('each question needs an option')
    return {'media': media, 'masks': masks, 'options': options, 'visible': visible,
            'question': np.stack([text_vector(text, 32) for text in frame.question])}, frame.qa_id.tolist()


def fit_normalizer(inputs, rows):
    normalizer = {}
    for index, name in enumerate(MODALITIES):
        selected = np.asarray(rows)[inputs['masks'][rows, index]]
        if not len(selected):
            raise ValueError('no available training rows for ' + name)
        values = inputs['media'][name][selected].astype(np.float64)
        normalizer[name] = {'mean': values.mean(axis=0).astype(np.float32),
                            'scale': np.maximum(values.std(axis=0), 1e-4).astype(np.float32),
                            'fit_rows': selected.tolist()}
    return normalizer


def tensors(inputs, normalizer):
    values = []
    for index, name in enumerate(MODALITIES):
        raw = inputs['media'][name]
        value = np.clip((raw - normalizer[name]['mean']) / normalizer[name]['scale'], -10, 10)
        value[~inputs['masks'][:, index]] = 0
        values.append(value)
    values.append(inputs['masks'].astype(np.float32))
    return {'media': torch.from_numpy(np.concatenate(values, axis=1)),
            **{name: torch.from_numpy(inputs[name]) for name in ('options', 'question', 'visible')}}


class LocalEmotionStudent(nn.Module):
    def __init__(self, media_width, hidden=64, dropout=.1):
        super().__init__()
        self.media = nn.Sequential(nn.Linear(media_width, hidden), nn.LayerNorm(hidden), nn.Tanh(), nn.Dropout(dropout))
        self.option = nn.Linear(128, hidden)
        self.question = nn.Linear(32, hidden, bias=False)
        self.bias = nn.Linear(128, 1)
        self.dropout = nn.Dropout(dropout)
        self.scale = hidden ** -.5

    def forward(self, media, options, question, visible):
        video = self.media(media)
        words = self.dropout(torch.tanh(self.option(options) + self.question(question)[:, None]))
        logits = torch.einsum('bh,bkh->bk', video, words) * self.scale + self.bias(options).squeeze(-1)
        return logits.masked_fill(~visible, -1e9)


def objective(logits, target, soft, *, temperature, distill_weight):
    ce = F.cross_entropy(logits, target)
    kl = logits.new_zeros(())
    if distill_weight:
        if soft is None or soft.shape != logits.shape:
            raise ValueError('distillation needs aligned teacher marginals')
        kl = F.kl_div(F.log_softmax(logits / temperature, dim=-1), soft, reduction='batchmean') * temperature ** 2
    return ce + distill_weight * kl, ce, kl


def infer(model, batch, rows):
    """Fixed one-question evaluation makes single/batch numerical paths identical."""
    model.eval()
    results = []
    with torch.no_grad():
        for row in rows:
            logits = model(**{key: value[int(row):int(row) + 1] for key, value in batch.items()})
            results.append(torch.softmax(logits, dim=-1)[0].numpy())
    return np.asarray(results, dtype=np.float32)


def state_digest(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()
