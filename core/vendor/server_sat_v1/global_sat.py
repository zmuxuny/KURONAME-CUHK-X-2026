"""Exact option-logic audit; labels are never used to construct clauses.
Cross-recording mode assumes the already-audited three-recording scene layout.
It is NOT a replacement for visual evidence or proof of test correctness.
"""
from __future__ import annotations
import itertools, json, re, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from pysat.solvers import Solver

LETTERS = 'ABCD'
CATEGORIES = ('single', 'multi', 'combination')

def atoms(value):
    return tuple(re.sub(r'\s+', ' ', p.strip().lower()) for p in str(value).split(',') if p.strip())

def clip_key(path):
    return re.sub(r'/(Depth|Depth_Color|IR|Thermal)/[^/]+\.mp4$', '', str(path), flags=re.I)

def prepare(frame):
    x = frame.copy()
    x['clip'] = x.path.map(clip_key)
    x['subject'] = x.path.str.extract(r'(user\d+)(?:/|$)', expand=False).fillna('test')
    x['scene'] = x['clip']
    hau = x.source.eq('HAU')
    clips = sorted(x.loc[hau, 'clip'].unique())
    if all(re.fullmatch(r'HAU/user\d+/.+-[123]', c) for c in clips):
        groups = {c: c.rsplit('-', 1)[0] for c in clips}
    else:
        ids = [int(re.search(r'LM_test_(\d+)$', c).group(1)) for c in clips]
        start = min(ids)
        if sorted(ids) != list(range(start, max(ids)+1)) or len(ids) % 3:
            raise ValueError('Test HAU layout is not an intact sequence of triplets')
        groups = {c: f'test_scene_{(i-start)//3:03d}' for c, i in zip(clips, ids)}
    x.loc[hau, 'scene'] = x.loc[hau, 'clip'].map(groups)
    return x

class Problem:
    def __init__(self, frame):
        self.frame = frame[frame.category.isin((*CATEGORIES, 'sequence'))].copy()
        self.clauses, self.variables, self.options = [], {}, {}
        for row in self.frame.itertuples():
            opts = []
            for letter in LETTERS:
                aa = atoms(getattr(row, letter))
                vv = [self.var(('action', a)) for a in aa]
                if len(vv) == 1:
                    y = vv[0]
                else:
                    y = self.var(('conjunction', *sorted(aa)))
                    self.clauses.extend([[-y, v] for v in vv])
                    self.clauses.append([y] + [-v for v in vv])
                self.options[(row.qa_id, letter)] = y
                opts.append(y)
            if row.category in ('single', 'combination'):
                self.clauses.append(opts)
                self.clauses.extend([[-a, -b] for a, b in itertools.combinations(opts, 2)])
            elif row.category == 'multi':
                self.clauses.append(opts)
            elif row.category == 'sequence':
                self.clauses.extend([[v] for v in opts])
        self.nvars = len(self.variables)

    def var(self, key):
        if key not in self.variables:
            self.variables[key] = len(self.variables) + 1
        return self.variables[key]

    def assumptions(self, predictions):
        out = []
        for row in self.frame.itertuples():
            if row.category not in CATEGORIES or row.qa_id not in predictions:
                continue
            answer = predictions[row.qa_id]
            out.extend(self.options[(row.qa_id, l)] * (1 if l in answer else -1) for l in LETTERS)
        return out

    def backbone(self):
        with Solver(name='g4', bootstrap_with=self.clauses) as solver:
            if not solver.solve():
                return None
            result = {}
            for v in set(self.options.values()):
                pos = solver.solve(assumptions=[v])
                neg = solver.solve(assumptions=[-v])
                result[v] = 1 if not neg else (0 if not pos else -1)
            return result

def audit(frame, scope, parent=None):
    records, groups = [], []
    for key, part in frame[frame.source.eq('HAU')].groupby(scope, sort=True):
        p = Problem(part)
        forced = p.backbone()
        truth = dict(zip(part.qa_id, part.answer)) if 'answer' in part else None
        with Solver(name='g4', bootstrap_with=p.clauses) as solver:
            truth_ok = solver.solve(assumptions=p.assumptions(truth)) if truth else None
            parent_ok = solver.solve(assumptions=p.assumptions(parent)) if parent else None
        groups.append(dict(group=key, scope=scope, nvars=p.nvars, nclauses=len(p.clauses),
                           satisfiable=forced is not None, truth_satisfiable=truth_ok, parent_satisfiable=parent_ok))
        for row in part[part.category.isin(CATEGORIES)].itertuples():
            flags = [forced[p.options[(row.qa_id,l)]] if forced is not None else -1 for l in LETTERS]
            positive = ''.join(l for l,f in zip(LETTERS,flags) if f == 1)
            negative = ''.join(l for l,f in zip(LETTERS,flags) if f == 0)
            complete = all(f != -1 for f in flags)
            old = str(parent[row.qa_id]) if parent else ''
            pred = ''.join(l for l in LETTERS if l in positive or (l in old and l not in negative))
            if row.category != 'multi' and len(pred) != 1:
                pred = old
            if not pred:
                pred = old
            ans = str(row.answer) if truth else ''
            valid = set(positive) <= set(ans) and not (set(negative) & set(ans)) if truth else None
            records.append(dict(qa_id=row.qa_id, subject=row.subject, category=row.category, clip=row.clip,
                group=key, positive=positive, negative=negative, complete=complete, forced_valid=valid,
                answer=ans, parent=old, prediction=pred, changed=pred != old if parent else False))
    return pd.DataFrame(records), pd.DataFrame(groups)

def main():
    root = Path('/data/cuhkx')
    train = prepare(pd.read_csv(root/'data/qa/training_qa.csv', keep_default_na=False))
    test = prepare(pd.read_csv(root/'data/qa/test_qa.csv', keep_default_na=False))
    baseline = pd.read_csv(root/'submissions/reference_56105805.csv', keep_default_na=False)
    parent = dict(zip(baseline.qa_id, baseline.prediction))
    out = root/'experiments/sat_audit_v1'
    out.mkdir(exist_ok=True)
    summary = {}
    for scope in ('clip', 'scene'):
        tr, tg = audit(train, scope)
        te, eg = audit(test, scope, parent)
        for name, df in [('train_rows',tr),('train_groups',tg),('test_rows',te),('test_groups',eg)]:
            df.to_csv(out/f'{scope}_{name}.csv',index=False)
        summary[scope] = dict(train_rows=len(tr), train_forced_rows=int(tr.complete.sum()),
            train_wrong_forced_rows=int((~tr.forced_valid).sum()),
            train_unsat_groups=int((~tg.satisfiable).sum()),
            train_truth_unsat_groups=int((~tg.truth_satisfiable).sum()),
            test_unsat_groups=int((~eg.satisfiable).sum()),
            test_parent_unsat_groups=int((~eg.parent_satisfiable).sum()),
            test_changes=int(te.changed.sum()), changes_by_category=te[te.changed].category.value_counts().to_dict())
        candidate = baseline.copy()
        changes = dict(zip(te.qa_id,te.prediction))
        candidate['prediction'] = [changes.get(q,p) for q,p in zip(candidate.qa_id,candidate.prediction)]
        candidate.to_csv(out/f'candidate_{scope}.csv',index=False,lineterminator='\n')
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)

if __name__ == '__main__': main()
