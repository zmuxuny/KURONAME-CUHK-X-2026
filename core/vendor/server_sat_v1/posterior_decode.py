"""Exact projected answer-state enumeration and whole-question Bayes decoding.
Unmentioned action assignments are NOT counted as extra answer-state mass.
"""
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
from pysat.solvers import Solver
from global_sat import Problem, prepare, CATEGORIES, LETTERS, audit
from frequency_model import decode

ROOT=Path('/data/cuhkx')

def enumerate_answers(part,max_states=4096):
    p=Problem(part)
    rows=p.frame[p.frame.category.isin(CATEGORIES)]
    qids=rows.qa_id.tolist()
    vars_=np.array([[p.options[(q,l)] for l in LETTERS] for q in qids],dtype=int)
    relevant=sorted(set(vars_.reshape(-1).tolist()))
    states=[]
    with Solver(name='g4',bootstrap_with=p.clauses) as solver:
        while solver.solve():
            model=set(v for v in solver.get_model() if v>0)
            states.append([[int(v in model) for v in vv] for vv in vars_])
            solver.add_clause([-v if v in model else v for v in relevant])
            if len(states)>max_states: return qids,None,'state_cap'
    return qids,np.asarray(states,dtype=np.int8),'ok' if states else 'unsatisfiable'

def decisions(qids,states,prob_by_q,temperature=1.0):
    prob=np.clip(np.stack([prob_by_q[q] for q in qids]),1e-6,1-1e-6)
    score=(states*np.log(prob)+(1-states)*np.log1p(-prob)).sum((1,2))/temperature
    weight=np.exp(score-score.max());weight/=weight.sum()
    masks=np.sum(states*(1<<np.arange(4))[None,None,:],axis=2)
    marginal=np.stack([np.bincount(masks[:,i],weights=weight,minlength=16) for i in range(len(qids))])
    expected=np.sum(np.stack([marginal[i,masks[:,i]] for i in range(len(qids))]),axis=0)
    map_index=int(score.argmax());mbr_index=int(expected.argmax())
    def letters(mask):return ''.join(l for i,l in enumerate(LETTERS) if int(mask)&(1<<i))
    result=[]
    for i,q in enumerate(qids):
        marginal_order=np.argsort(marginal[i])
        result.append(dict(qa_id=q,map=letters(masks[map_index,i]),
            mbr=letters(masks[mbr_index,i]),marginal=letters(marginal_order[-1]),
            map_prob=float(marginal[i,masks[map_index,i]]),mbr_prob=float(marginal[i,masks[mbr_index,i]]),
            marginal_prob=float(marginal[i,marginal_order[-1]]),
            marginal_margin=float(marginal[i,marginal_order[-1]]-marginal[i,marginal_order[-2]]),
            nstates=len(states),joint_entropy=float(-(weight*np.log(weight+1e-30)).sum())))
    return result

def prob_dict(meta,prob):
    return {q:g.probability.to_numpy() for q,g in meta.assign(probability=prob).groupby('qa_id',sort=False)}

def run_split(frame,meta,prob,scope,temperature=1.0):
    by_q=prob_dict(meta,prob)
    independent=decode(meta,prob).set_index('qa_id').prediction.to_dict()
    predictions=[]; groups=[]
    for key,part in frame[frame.source.eq('HAU')].groupby(scope,sort=True):
        qids,states,status=enumerate_answers(part)
        groups.append(dict(group=key,status=status,nstates=len(states) if states is not None else -1))
        if status=='ok':
            predictions.extend(decisions(qids,states,by_q,temperature))
        elif scope=='scene':
            for _,clip in part.groupby('clip'):
                cq,cs,cstatus=enumerate_answers(clip)
                if cstatus=='ok': predictions.extend(decisions(cq,cs,by_q,temperature))
        else:
            for q in qids:
                old=independent[q]
                predictions.append(dict(qa_id=q,map=old,mbr=old,marginal=old,map_prob=0.,mbr_prob=0.,
                    marginal_prob=0.,marginal_margin=0.,nstates=0,joint_entropy=0.))
    ans=pd.DataFrame(predictions)
    keep=['qa_id','category','subject','clip','scene']+(['answer'] if 'answer' in frame else [])
    ans=frame[frame.source.eq('HAU') & frame.category.isin(CATEGORIES)][keep].merge(ans,on='qa_id',validate='one_to_one')
    assert len(ans)==len(meta)//4
    ans['independent']=ans.qa_id.map(independent)
    return ans,pd.DataFrame(groups)

def main():
    f=ROOT/'experiments/frequency_loso_v1';out=ROOT/'experiments/posterior_v1';out.mkdir(exist_ok=True)
    train=prepare(pd.read_csv(ROOT/'data/qa/training_qa.csv',keep_default_na=False))
    test=prepare(pd.read_csv(ROOT/'data/qa/test_qa.csv',keep_default_na=False))
    tm=pd.read_csv(f/'train_meta.csv');em=pd.read_csv(f/'test_meta.csv')
    prob=np.load(f/'probabilities.npz');summary={};start=time.time()
    for scope in ['clip','scene']:
        tr,tg=run_split(train,tm,prob['oof'].mean(1),scope)
        te,eg=run_split(test,em,prob['test'].mean(1),scope)
        tr.to_csv(out/f'{scope}_train.csv',index=False);te.to_csv(out/f'{scope}_test.csv',index=False)
        tg.to_csv(out/f'{scope}_train_groups.csv',index=False);eg.to_csv(out/f'{scope}_test_groups.csv',index=False)
        metrics={}
        for method in ['independent','map','mbr','marginal']:
            good=tr[method].eq(tr.answer);parent_good=tr.independent.eq(tr.answer)
            metrics[method]=dict(correct=int(good.sum()),total=len(tr),fixes=int((good&~parent_good).sum()),
                harms=int((~good&parent_good).sum()),by_category=tr.assign(good=good).groupby('category').good.sum().to_dict())
        both=tr['map'].eq(tr.answer);mbr=tr.mbr.eq(tr.answer)
        metrics['mbr_vs_map']=dict(changed=int(tr.mbr.ne(tr['map']).sum()),fixes=int((mbr&~both).sum()),harms=int((both&~mbr).sum()))
        metrics['group_status_train']=tg.status.value_counts().to_dict();metrics['group_status_test']=eg.status.value_counts().to_dict()
        summary[scope]=metrics;print(scope,json.dumps(metrics,indent=2),flush=True)
    summary['seconds']=time.time()-start
    (out/'summary.json').write_text(json.dumps(summary,indent=2))

if __name__=='__main__': main()
