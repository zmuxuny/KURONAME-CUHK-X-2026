"""Reimplementation of attached option_frequency.py's 210-D answer-free features.
Train-only ExtraTrees; subject-exclusive OOF; fixed three-leaf ensemble.
"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import hashlib, json, time
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from global_sat import prepare, atoms, CATEGORIES, LETTERS

ROOT=Path('/data/cuhkx')

def features(frame):
    hau=frame[frame.source.eq('HAU')]
    occurrences=defaultdict(list)
    for r in hau[~hau.category.eq('emotion')].itertuples():
        for l in LETTERS:
            for a in set(atoms(getattr(r,l))):
                occurrences[(r.scene,a)].append((r.clip,r.qa_id,r.category))
    xx,meta=[],[]
    for r in hau[hau.category.isin(CATEGORIES)].itertuples():
        raw=[]
        for l in LETTERS:
            aa=atoms(getattr(r,l)); af=[]
            for a in aa:
                items=occurrences[(r.scene,a)]
                sibling=[v for v in items if v[0]!=r.clip]
                same=[v for v in items if v[0]==r.clip and v[1]!=r.qa_id]
                values=[len(sibling),len({v[1] for v in sibling}),len({v[0] for v in sibling}),len(items),len(same)]
                for cat in (*CATEGORIES,'sequence'):
                    values += [sum(v[2]==cat for v in sibling),len({v[1] for v in sibling if v[2]==cat}),sum(v[2]==cat for v in same)]
                af.append(values)
            af=np.asarray(af,dtype=np.float32)
            raw.append(np.concatenate([af.min(0),af.mean(0),af.max(0),af.sum(0),[len(aa)]]))
        raw=np.asarray(raw,dtype=np.float32)
        for i,l in enumerate(LETTERS):
            xx.append(np.concatenate([raw[i],(raw<raw[i]).sum(0),raw[i]-raw.mean(0),[r.category==c for c in CATEGORIES]]))
            m=dict(qa_id=r.qa_id,clip=r.clip,scene=r.scene,category=r.category,subject=r.subject,letter=l)
            if 'answer' in frame.columns:
                m.update(answer=r.answer,label=int(l in r.answer))
            meta.append(m)
    x=np.asarray(xx,dtype=np.float32)
    assert x.shape[1]==210 and np.isfinite(x).all()
    return x,pd.DataFrame(meta)

def decode(meta,probabilities):
    out=[]
    for q,g in meta.assign(probability=probabilities).groupby('qa_id',sort=False):
        p=g.probability.to_numpy();r=g.iloc[0]
        pred=''.join(l for l,v in zip(LETTERS,p) if v>=.475) if r.category=='multi' else ''
        if not pred: pred=LETTERS[int(p.argmax())]
        out.append(dict(qa_id=q,category=r.category,subject=r.subject,prediction=pred,answer=r.get('answer','')))
    return pd.DataFrame(out)

def main():
    out=ROOT/'experiments/frequency_loso_v1';out.mkdir(exist_ok=True)
    models=ROOT/'models/frequency_loso_v1';models.mkdir(exist_ok=True)
    tr=prepare(pd.read_csv(ROOT/'data/qa/training_qa.csv',keep_default_na=False))
    te=prepare(pd.read_csv(ROOT/'data/qa/test_qa.csv',keep_default_na=False))
    x,m=features(tr);xt,mt=features(te);m.to_csv(out/'train_meta.csv',index=False);mt.to_csv(out/'test_meta.csv',index=False)
    subjects=sorted(m.subject.unique(),key=lambda s:int(s.replace('user','')))
    y=m.label.to_numpy();group=m.subject.to_numpy();leafs=[2,5,10]
    oof=np.full((len(m),len(leafs)),np.nan,dtype=np.float32)
    test_probs=np.zeros((len(mt),len(leafs)),dtype=np.float32)
    timings=[];t0=time.time()
    for k,leaf in enumerate(leafs):
        for subject in subjects+['final']:
            t=time.time();training=group!=subject;testing=group==subject
            model=ExtraTreesClassifier(n_estimators=200,min_samples_leaf=leaf,max_features='sqrt',
                class_weight='balanced',n_jobs=8,random_state=20260912+1000*k)
            model.fit(x[training],y[training])
            if subject=='final':
                test_probs[:,k]=model.predict_proba(xt)[:,1]
                joblib.dump(model,models/f'leaf{leaf}_final.joblib',compress=3)
            else:
                oof[testing,k]=model.predict_proba(x[testing])[:,1]
            entry=dict(leaf=leaf,heldout=subject,n_train=int(training.sum()),seconds=round(time.time()-t,3))
            timings.append(entry);print(json.dumps(entry),flush=True)
        np.savez_compressed(out/'probabilities.npz',oof=oof,test=test_probs)
        pd.DataFrame(timings).to_csv(out/'timings.csv',index=False)
    assert np.isfinite(oof).all()
    pred=decode(m,oof.mean(1));pred.to_csv(out/'oof_independent.csv',index=False)
    pred_test=decode(mt,test_probs.mean(1));pred_test.to_csv(out/'test_independent.csv',index=False)
    summary=dict(n_subjects=len(subjects),n_questions=len(pred),correct=int(pred.prediction.eq(pred.answer).sum()),
        by_category=pred.assign(correct=pred.prediction.eq(pred.answer)).groupby('category').correct.agg(['sum','count']).to_dict('index'),
        seconds=round(time.time()-t0,2),trees=200,leafs=leafs,
        selection='Fixed equal ensemble, all 18 leave-one-subject-out folds; no inner hyperparameter selection',
        boundary='Not an OOF reproduction of the full 0.91520 reference; metadata-only matching control',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__': main()
