#!/usr/bin/env python3
"""Frozen RGB625 HAU head from current media, without a saved Test feature bank."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import cv2,joblib,numpy as np,torch
from cuhkx.harn_tail import configure_torch
from cuhkx.r34_video import metadata_arrays
from cuhkx.sequence_r34_transfer import select_views
from scripts.extract_r34_source_hau import models,tensors,forward

def main():
 p=argparse.ArgumentParser()
 for x in ['manifest','media-root','rois','weights','source-model','head','output']:p.add_argument('--'+x,type=Path,required=True)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 configure_torch();torch.set_num_threads(2);cv2.setNumThreads(1)
 cfg={'weights':str(a.weights),'sources':{'rgb':str(a.source_model)},'data_root':str(a.media_root),'rois':str(a.rois)}
 networks,identities=models(cfg,arms=('rgb',));head=joblib.load(a.head)
 raw=json.loads(a.manifest.read_text());metadata=metadata_arrays(raw)
 lookup={(str(u),str(m),int(w)):i for i,(u,m,w) in enumerate(zip(metadata['units'],metadata['modalities'],metadata['window_indices']))}
 grouped={}
 for r in raw['records']:grouped.setdefault(r['unit'],{})[r['modality']]=r
 units=sorted(grouped);values=[]
 for i,u in enumerate(units):
  _,mask,times,reason=select_views(metadata,lookup,u,allow_duplicate_windows=True)
  if mask is None:raise ValueError('No available HAU view: '+u)
  samples,_=tensors(cfg,u,grouped[u],mask);features=np.zeros((8,2,512),np.float32)
  for view,x in samples.items():features[:,view]=forward(networks['rgb'],x)
  X=np.concatenate([features.reshape(1,-1),mask.reshape(1,-1)],axis=1)
  values.append(head['presence'].predict(X)[0])
  (a.output/'progress.json').write_text(json.dumps({'completed':i+1,'required':len(units)})+'\n')
  print('source HAU',i+1,'/',len(units),flush=True)
 np.savez_compressed(a.output/'probabilities.npz',units=np.asarray(units),probabilities=np.asarray(values),classes=np.arange(40))
 (a.output/'report.json').write_text(json.dumps({'units':len(units),'source':identities,'head':str(a.head),'new_fits':0,'precision':'BF16 encoder / FP32 pooled512; frozen final head'})+'\n')
if __name__=='__main__':main()
