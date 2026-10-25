"""Whole-question minimum-edit repair, with Train-only option scores for ties.
Every same-clip action question participates. No selected test-row rules.
"""
import json,time
from pathlib import Path
import numpy as np
import pandas as pd
from global_sat import prepare,audit,CATEGORIES,LETTERS
from frequency_model import decode
from posterior_decode import enumerate_answers,prob_dict
ROOT=Path('/data/cuhkx')

def repair(frame,meta,prob,parent):
    pq=prob_dict(meta,prob);out=[]
    for clip,part in frame[frame.source.eq('HAU')].groupby('clip',sort=True):
        qs,states,status=enumerate_answers(part)
        if status!='ok': raise ValueError(f'No complete candidate enumeration: {clip}')
        masks=(states*(1<<np.arange(4))[None,None,:]).sum(2)
        old=np.array([sum(1<<i for i,l in enumerate(LETTERS) if l in parent[q]) for q in qs])
        edits=(masks!=old[None,:]).sum(1)
        eligible=edits==edits.min()
        pp=np.clip(np.stack([pq[q] for q in qs]),1e-6,1-1e-6)
        scores=(states*np.log(pp)+(1-states)*np.log1p(-pp)).sum((1,2))
        selected=int(np.argmax(np.where(eligible,scores,-np.inf)))
        local=np.flatnonzero(eligible);w=np.exp(scores[local]-scores[local].max());w/=w.sum()
        chosen=[]
        for i,q in enumerate(qs):
            chosen.append(''.join(l for k,l in enumerate(LETTERS) if masks[selected,i]&(1<<k)))
        for i,q in enumerate(qs):
            agreement=float(w[masks[local,i]==masks[selected,i]].sum())
            out.append(dict(qa_id=q,parent=parent[q],prediction=chosen[i],changed=chosen[i]!=parent[q],
                clip=clip,min_question_edits=int(edits.min()),n_optimal_states=int(eligible.sum()),
                n_all_states=len(states),restricted_confidence=agreement))
    return pd.DataFrame(out)

def main():
    out=ROOT/'experiments/repair_v1';out.mkdir(exist_ok=True)
    f=ROOT/'experiments/frequency_loso_v1';p=np.load(f/'probabilities.npz')
    tr=prepare(pd.read_csv(ROOT/'data/qa/training_qa.csv',keep_default_na=False))
    te=prepare(pd.read_csv(ROOT/'data/qa/test_qa.csv',keep_default_na=False))
    tm=pd.read_csv(f/'train_meta.csv');em=pd.read_csv(f/'test_meta.csv')
    oldtrain=decode(tm,p['oof'].mean(1));parent=dict(zip(oldtrain.qa_id,oldtrain.prediction))
    train=repair(tr,tm,p['oof'].mean(1),parent)
    train=train.merge(tr[['qa_id','answer','category','subject']],on='qa_id',validate='one_to_one')
    testbase=pd.read_csv(ROOT/'submissions/reference_56105805.csv',keep_default_na=False)
    test=repair(te,em,p['test'].mean(1),dict(zip(testbase.qa_id,testbase.prediction)))
    test=test.merge(te[['qa_id','category']],on='qa_id',validate='one_to_one')
    train.to_csv(out/'train_repair.csv',index=False);test.to_csv(out/'test_repair.csv',index=False)
    certain,_=audit(tr,'clip',parent)
    truth=tr.set_index('qa_id').answer.to_dict();certain['answer']=certain.qa_id.map(truth)
    summary={}
    for name,df in [('min_edit_repair',train),('backbone_only',certain)]:
        good=df.prediction.eq(df.answer);pgood=df.parent.eq(df.answer)
        summary[name]=dict(correct=int(good.sum()),parent_correct=int(pgood.sum()),
            changed=int(df.prediction.ne(df.parent).sum()),fixes=int((good&~pgood).sum()),harms=int((~good&pgood).sum()))
    summary['test_changes']=int(test.changed.sum())
    summary['test_conflicting_clips']=int(test.loc[test.min_question_edits.gt(0),'clip'].nunique())
    candidate=testbase.copy();overlay=test.set_index('qa_id').prediction.to_dict()
    candidate['prediction']=[overlay.get(q,pred) for q,pred in zip(candidate.qa_id,candidate.prediction)]
    assert candidate.qa_id.tolist()==te.qa_id.tolist() and len(candidate)==682
    candidate.to_csv(out/'candidate_repair.csv',index=False,lineterminator='\n')
    _,consistency=audit(te,'clip',dict(zip(candidate.qa_id,candidate.prediction)))
    summary['repaired_test_inconsistent_clips']=int((~consistency.parent_satisfiable).sum())
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__': main()
