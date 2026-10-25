"""Frozen second Selected submission: paired HARn option fusion and temporal residual."""
from pathlib import Path
import json,gc
import numpy as np,pandas as pd,torch,joblib
from scipy.special import expit

def load_npz(path):
 with np.load(path) as z:return {k:z[k].copy() for k in z.files}

def apply_secondary(package,work,qa,media,parent,run):
 from scripts.extract_r34_midfeatures import collect_inputs
 from cuhkx.metadata import feature_unit
 cfg=json.loads((package/'configs/submission_56258762.json').read_text())
 core=package/'core';base=work/'baseline';state=json.loads((base/'raw/state.json').read_text())
 crops=json.loads(Path(state['stages']['roi']['primary_output']).read_text())['windows']
 out=work/'secondary';out.mkdir()
 harn=qa.loc[qa.source.eq('HARn')]
 if len(harn):
  from cuhkx.harn_tail import configure_torch
  from cuhkx.r34_adaptation import restore
  from cuhkx.r34_video import sample_preprocessed
  from scripts.evaluate_r34_adaptation import probabilities
  from scripts.evaluate_r34_expert_slot import replace_slot,half_slot
  from cuhkx.harn_seed_ensemble import equal_evidence
  from cuhkx.harn_three_family import three_family_evidence
  from cuhkx.harn_option_fusion import option_fusion
  from cuhkx.selection_harn_decode import decode_questions,QA_COLUMNS
  units=sorted(set(harn.path.map(lambda p:feature_unit(p,training=False))))
  records=collect_inputs(media/'visual',[{'unit':u,'source':'HARn'} for u in units],crops)['records']
  grouped={u:[r for r in records if r['unit']==u and r['available']] for u in units}
  settings={'weights':str(core/json.loads((core/'configs/r34_midfeatures_v1.json').read_text())['weights']),'training':{'seed':cfg['target_seed']}}
  configure_torch();model,payload=restore(settings,package/cfg['target_model']);classes=payload['supported_classes']
  matrix=np.zeros((len(units),44),np.float32);present=np.asarray([bool(grouped[u]) for u in units])
  for start in range(0,len(units),16):
   chosen=units[start:start+16];tensors=[];owners=[]
   for i,u in enumerate(chosen):
    for r in grouped[u]:tensors.append(sample_preprocessed(r));owners.append(i)
   if tensors:
    values=probabilities(model,torch.cat(tensors),classes);owners=np.asarray(owners)
    for i,u in enumerate(chosen):
     if grouped[u]:matrix[start+i]=values[owners==i].mean(0)
  np.savez_compressed(out/'target_probabilities.npz',units=np.asarray(units),probabilities=matrix,present=present,classes=np.asarray(classes))
  del model;gc.collect();torch.cuda.empty_cache()
  hb=base/'inference/four_expert_parent.work/harn'
  raw={n:load_npz(hb/(n+'_probabilities.npz')) for n in ['original','repeated','r34','videomae']}
  r18=equal_evidence(raw['original'],raw['repeated']);new=replace_slot(raw['r34'],np.asarray(units)[present],matrix[present],classes)
  fused=three_family_evidence(r18,half_slot(raw['r34'],new),raw['videomae'])
  profile=json.loads((base/'inference/profile.json').read_text());descriptor=json.loads(Path(profile['grouped']['checkpoints']['harn']).read_text())
  pdsc=json.loads(Path(descriptor['parent_checkpoint']['path']).read_text());dictionary=joblib.load(pdsc['classifiers']['original']['path'])
  target=harn[QA_COLUMNS];vectors=option_fusion(target,[[('r18',r18)],[('old_r34',raw['r34']),('aligned',new)],[('videomae',raw['videomae'])]],dictionary,training_paths=False)[0]
  pred=decode_questions(target,fused,dictionary,descriptor['selected_skeleton_weight'],training_paths=False,visual_option_scores=vectors)
  pred.to_csv(out/'harn_predictions.csv',index=False);p=pred.set_index('qa_id');parent.loc[p.index,'prediction']=p.prediction
 seq=qa.loc[qa.source.eq('HAU')&qa.category.eq('sequence')]
 if len(seq):
  from cuhkx.videomae_legacy_weights import load_runtime
  from cuhkx.videomae_windows import read_windows
  from cuhkx.generic_video_windows import window_timing
  from cuhkx.server_temporal_d1_sequence import d1_window_curves
  from cuhkx.server_temporal_d6_residual import features
  from cuhkx.server_temporal_d10_backoff import predict as prior_predict
  from cuhkx.temporal_consumer_residual import predict
  from cuhkx.sequence_skeleton_nested import prepare_sequences
  evidence=load_npz(base/'inference/d10/pair_scores.npz');frame=prepare_sequences(qa,training=False).set_index('qa_id').loc[evidence['qa_ids']]
  units=sorted(set(frame.unit));raw=collect_inputs(media/'visual',[{'unit':u,'source':'HAU'} for u in units],crops)
  settings=json.loads((package/'configs/secondary_d1_producer.json').read_text());settings['model_directory']=str(core/'pretrained/videomae')
  runtime=load_runtime(settings);_,model,processor=runtime;centers={};values=np.zeros((len(units),16,2,768),np.float32);mask=np.zeros((len(units),16,2),bool)
  for r in raw['records']:
   if not r['available']:continue
   i=units.index(r['unit']);v=['Depth_Color','IR'].index(r['modality'])
   windows=read_windows(r,settings)
   for k,frames in enumerate(windows):
    pixels=processor(list(frames),return_tensors='pt')['pixel_values'].to(device='cuda',dtype=torch.float16)
    with torch.inference_mode():
     tokens=model.videomae(pixels).last_hidden_state
     coarse=tokens.float().reshape(1,8,196,768).mean(2).mean(1)
     values[i,k,v]=model.fc_norm(coarse.half()).float().cpu().numpy()[0]
   mask[i,:,v]=True
   indices=window_timing(r['metadata'],settings)['frame_indices']
   centers[(r['unit'],r['modality'])]=(indices.reshape(16,8,2).mean(-1)/(r['metadata']['frame_count']-1)).mean(1)
  del runtime,model,processor;gc.collect();torch.cuda.empty_cache()
  matrix=np.concatenate([values.reshape(len(units),-1),mask.reshape(len(units),-1)],axis=1)
  matrix=matrix[[units.index(u) for u in frame.unit]]
  bundle=joblib.load(package/cfg['d1_model']);curves,valid=d1_window_curves(bundle['presence'],matrix,frame.unit.tolist(),centers)
  prior=json.loads((package/cfg['d10_model']).read_text());prob=evidence['pooled'];classes=evidence['classes'];logits=evidence['prior_logits']
  _,offset,_,_=prior_predict(prior,prob,logits,classes)
  result,scores,delta=predict(load_npz(package/cfg['temporal_residual']),features(prob,classes,curves,valid),prob,offset)
  pd.DataFrame({'qa_id':evidence['qa_ids'],'prediction':result}).to_csv(out/'temporal.csv',index=False)
  parent.loc[evidence['qa_ids'],'prediction']=result
  np.savez_compressed(out/'temporal_evidence.npz',qa_ids=evidence['qa_ids'],curves=curves,mask=valid,prediction=result,scores=scores)
 return parent
