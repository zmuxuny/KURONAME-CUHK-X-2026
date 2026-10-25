"""Unchanged frozen per-view D6 projection; extracted from archived run_d6_residual.py."""
import numpy as np
import torch
from scipy.special import expit
from cuhkx.server_temporal_d1_head import unpack_d1, MODALITIES

def view_curves(head,matrix,units,centers):
 f,m=unpack_d1(matrix,head.bins);z=head._standardized(f,m)
 with torch.inference_mode():
  h=torch.nn.functional.gelu(torch.nn.functional.linear(torch.from_numpy(z),torch.from_numpy(head.state['0.weight']),torch.from_numpy(head.state['0.bias'])))
  a=torch.nn.functional.linear(h,torch.from_numpy(head.state['3.weight']),torch.from_numpy(head.state['3.bias'])).numpy()
 p=expit(a.astype(float)-np.log(head.positive_weight));p[...,~head.supported]=.5
 dest=np.zeros((len(units),2,128,40),float);valid=m.any(1);grid=np.linspace(0,1,128)
 for i,u in enumerate(units):
  for v,mod in enumerate(MODALITIES):
   if not valid[i,v]:continue
   t=np.asarray(centers[(str(u),mod)],float)[m[i,:,v]];q=p[i,m[i,:,v],v];unique=np.unique(t)
   values=np.stack([q[t==k].mean(0) for k in unique])
   dest[i,v]=np.stack([np.interp(grid,unique,values[:,j]) for j in range(40)],1)
 return dest,valid
