#!/usr/bin/env python3
"""Official two-argument adapter for the two frozen Selected solutions."""
import argparse,json,os,re,shutil,subprocess,sys,time
from pathlib import Path
PACKAGE=Path(__file__).resolve().parents[1]
CORE=PACKAGE/'core';sys.path[:0]=[str(CORE),str(CORE/'src')]
import pandas as pd
from cuhkx.metadata import clip_key,feature_unit
from scripts.extract_r34_midfeatures import collect_inputs,write_json

def run(stage,args,work):
 print(stage,flush=True)
 with (work/(stage+'.log')).open('w') as log:
  subprocess.run(list(map(str,args)),cwd=CORE,stdout=log,stderr=subprocess.STDOUT,check=True)

def input_adapter(data,work):
 candidates=[data/'test_qa.csv',data/'Testing/test_qa.csv',data/'kaggle/test_qa.csv']
 qa_path=next((p for p in candidates if p.is_file()),None)
 if qa_path is None:raise FileNotFoundError('test_qa.csv is required in the supplied data directory')
 qa=pd.read_csv(qa_path,dtype=str,keep_default_na=False)
 qa=qa[[c for c in ['qa_id','source','path','category','question','A','B','C','D'] if c in qa]]
 q=work/'inputs/qa';q.mkdir(parents=True)
 qa.to_csv(q/'test_qa.csv',index=False)
 sample=qa_path.with_name('sample_submission.csv')
 if sample.exists():shutil.copyfile(sample,q/'sample_submission.csv')
 else:pd.DataFrame({'qa_id':qa.qa_id,'prediction':''}).to_csv(q/'sample_submission.csv',index=False)
 first=feature_unit(qa.iloc[0].path,training=False)
 # Discover a published Testing tree, optionally wrapped by its extracted archive directory.
 dirs=[data,*[p for p in data.iterdir() if p.is_dir()]]
 dirs += [c for p in dirs[1:] for c in p.iterdir() if c.is_dir() and c.name!='Training']
 visual=next((p for p in dirs if (p/first/'IR/IR.mp4').is_file()),None)
 if visual is None:
  clip=clip_key(qa.iloc[0].path)
  visual=next((p.parent for p in dirs if p.name=='Testing' and (p/clip/'IR/IR.mp4').is_file()),None)
 if visual is None:raise FileNotFoundError('No Testing visual directory matches the QA media paths')
 media=work/'inputs/media';media.mkdir()
 (media/'visual').symlink_to(visual,target_is_directory=True)
 sensor=next((p for p in dirs if p.name=='LMT_(IMU,Radar,Skeleton)' and (p/'Testing').is_dir()),None)
 if sensor is None:
  sensor=next((p for p in dirs if (p/first/'Skeleton').is_dir() and p!=visual),None)
 if sensor is None and (visual/first/'Skeleton').is_dir():sensor=visual
 if sensor is None:raise FileNotFoundError('No matching official nonvisual Testing directory')
 (media/'nonvisual').mkdir();(media/'nonvisual/LMT_(IMU,Radar,Skeleton)').symlink_to(sensor,target_is_directory=True)
 hau=qa.loc[qa.source.eq('HAU')];acq=None
 if len(hau):
  acq=q/'acquisition.csv';supplied=qa_path.with_name('acquisition.csv')
  if supplied.exists():shutil.copyfile(supplied,acq)
  else:
   clips=sorted(set(hau.path.map(clip_key)))
   if all(re.fullmatch(r'HAU/user\d+/.+-[123]',c) for c in clips):
    rows=[{'clip':c,'group':c.rsplit('-',1)[0],'position':int(c[-1])-1} for c in clips]
   else:
    ids={c:int(re.search(r'LM_test_(\d+)$',c).group(1)) for c in clips}
    start=min(ids.values())
    if sorted(ids.values())!=list(range(start,max(ids.values())+1)) or len(ids)%3:
     raise ValueError('The frozen triplet method requires complete published HAU triplets or supplied acquisition.csv (clip,group,position)')
    rows=[{'clip':c,'group':f'test_scene_{(i-start)//3:03d}','position':(i-start)%3} for c,i in ids.items()]
   pd.DataFrame(rows).to_csv(acq,index=False)
 return qa,q,media,acq

def main():
 p=argparse.ArgumentParser();p.add_argument('data_dir',type=Path);p.add_argument('output_csv',type=Path);p.add_argument('submission_id',nargs='?',default='56239668',choices=['56239668','56258762']);a=p.parse_args()
 output=a.output_csv.resolve();output.parent.mkdir(parents=True,exist_ok=True)
 work=output.with_suffix(output.suffix+'.work');work.mkdir(exist_ok=False)
 started=time.monotonic();qa,q,media,acq=input_adapter(a.data_dir.resolve(),work)
 baseline=[sys.executable,CORE/'scripts/run_raw_conditioned_package.py','--qa-dir',q,'--media-root',media,'--output',work/'baseline','--sat-python',sys.executable,'--cuda-slot']
 if acq:baseline+=['--acquisition',acq]
 run('baseline',baseline,work)
 finish(work,qa,q,media,acq,a.submission_id,output,started)

def finish(work,qa,q,media,acq,sid,output,started):
 baseline=work/'baseline';raw=baseline/'raw';inf=baseline/'inference'
 state=json.loads((raw/'state.json').read_text());features=state['features']
 parent_path=inf/'decision/composed_parent.csv' if acq else inf/'four_expert_parent.csv'
 parent=pd.read_csv(parent_path,dtype=str)
 if acq and (inf/'question_aliases.json').exists():
  from cuhkx.question_equivalence import expand_question_predictions
  parent=expand_question_predictions(parent,json.loads((inf/'question_aliases.json').read_text()))
 parent=parent.set_index('qa_id')
 active=qa.loc[qa.source.eq('HAU')&qa.category.isin(['single','multi','combination'])]
 if len(active):
  cfg=json.loads((PACKAGE/'configs/submission_56239668.json').read_text())
  units=sorted(set(active.path.map(lambda p:feature_unit(p,training=False))))
  crops=json.loads(Path(state['stages']['roi']['primary_output']).read_text())['windows']
  collected=collect_inputs(media/'visual',[{'unit':u,'source':'HAU'} for u in units],crops)
  manifest=work/'source_inputs.json';write_json(manifest,collected)
  import hashlib
  sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
  yolo=CORE/'pretrained/kuno/yolo11n.pt'
  plan=work/'source_roi_plan.json';write_json(plan,{'manifest':str(manifest),'manifest_sha256':sha(manifest),'weights':str(yolo),'weights_sha256':sha(yolo),'data_root':str(media/'visual')})
  run('source_rois',[sys.executable,CORE/'scripts/produce_r34_window_rois.py','--plan',plan,'--output',work/'source_rois','--split','Testing'],work)
  weights=CORE/json.loads((CORE/'configs/r34_midfeatures_v1.json').read_text())['weights']
  run('source_presence',[sys.executable,CORE/'scripts/selection_source_presence.py','--manifest',manifest,'--media-root',media/'visual','--rois',work/'source_rois','--weights',weights,'--source-model',PACKAGE/cfg['source_assets']['source_rgb625'],'--head',PACKAGE/cfg['source_assets']['source_hau_head'],'--output',work/'source_presence'],work)
  profile=json.loads((inf/'profile.json').read_text());ck=profile['grouped']['checkpoints']['action']
  command=[sys.executable,CORE/'scripts/selection_action.py','--source-presence',work/'source_presence/probabilities.npz','infer','--checkpoint',ck,'--qa-csv',q/'test_qa.csv','--logits',features['logits8'],features['logits16'],'--window-features',*[features[x] for x in ['windows16','windows32','windows48']],'--scene-manifest',acq,'--workers','2','--output-dir',work/'selected_action']
  run('selected_action',command,work)
  pred=pd.read_csv(work/'selected_action/test_predictions.csv',dtype=str).set_index('qa_id')
  parent.loc[pred.index,'prediction']=pred.prediction
 if sid=='56258762':
  from selection_secondary import apply_secondary
  parent=apply_secondary(PACKAGE,work,qa,media,parent,run)
 composed=work/'composed_parent.csv';parent.loc[qa.qa_id].reset_index().to_csv(composed,index=False)
 if acq:
  groups=work/'groups.csv';pd.read_csv(acq)[['clip','group']].to_csv(groups,index=False)
  cmd=[sys.executable,CORE/'scripts/apply_explicit_group_sat.py','--qa-csv',q/'test_qa.csv','--parent-csv',composed,'--profile',inf/'decision/runtime/sat_profile.json','--scene-manifest',groups,'--output-dir',work/'sat']
 else:
  cmd=[sys.executable,CORE/'scripts/apply_scored_sat_v3.py','--qa-csv',q/'test_qa.csv','--parent-csv',composed,'--profile',CORE/'configs/sat_independent_general_v1.json','--output-dir',work/'sat']
 run('selected_sat',cmd,work);shutil.copyfile(work/'sat/submission.csv',output)
 write_json(work/'reproduction.json',{'submission_id':sid,'rows':len(qa),'elapsed_seconds':time.monotonic()-started,'new_fits':0,'input_root':str(media),'output':str(output)})
 print('Finished',sid,output,flush=True)
if __name__=='__main__':main()
