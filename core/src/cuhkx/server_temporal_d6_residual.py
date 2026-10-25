"""D6 outer-safe offset sequence likelihood on frozen strong and view cues."""
from __future__ import annotations

import itertools

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp

PAIRS=tuple(itertools.combinations(range(4),2))
ORDERS=tuple(itertools.permutations(range(4)))
ANSWERS=np.asarray([''.join('ABCD'[j] for j in order) for order in ORDERS])
BITS=np.asarray([[order.index(a)<order.index(b) for a,b in PAIRS] for order in ORDERS],float)
def logit(p):
 p=np.clip(np.asarray(p,float),1e-5,1-1e-5)
 return np.log(p)-np.log1p(-p)
def view_pair_cues(left,right):
 left,right=np.asarray(left,float),np.asarray(right,float)
 if left.shape!=right.shape or left.ndim!=1 or not len(left):raise ValueError('curve shape')
 if not np.isfinite(left).all() or not np.isfinite(right).all() or min(left.min(),right.min())<0:raise ValueError('curve values')
 if left.sum()<=1e-12 or right.sum()<=1e-12:return np.zeros(7,float)
 a,b=left/left.sum(),right/right.sum();t=np.linspace(0,1,len(a))
 ma=t[min(np.searchsorted(np.cumsum(a),.5),len(t)-1)]
 mb=t[min(np.searchsorted(np.cumsum(b),.5),len(t)-1)]
 before=np.sum(a*(1-np.cumsum(b)));after=np.sum(b*(1-np.cumsum(a)))
 ea=-np.sum(a*np.log(np.maximum(a,1e-12)));eb=-np.sum(b*np.log(np.maximum(b,1e-12)))
 return np.array([np.dot(t,b-a),mb-ma,t[b.argmax()]-t[a.argmax()],before-after,
  np.log(max(left.max(),1e-12)/max(right.max(),1e-12)),
  np.log(max(left.mean(),1e-12)/max(right.mean(),1e-12)),ea-eb],float)
def features(probability,classes,curves,available):
 p=np.asarray(probability,float)
 if p.ndim!=2 or p.shape[1]!=6 or curves.shape!=(len(p),2,128,40):raise ValueError('evidence shape')
 if classes.shape!=(len(p),4) or available.shape!=(len(p),2):raise ValueError('identity/mask shape')
 out=np.zeros((len(p),6,15),float);out[:,:,0]=logit(p)
 for i in range(len(p)):
  for j,(a,b) in enumerate(PAIRS):
   for v in range(2):
    if available[i,v]:out[i,j,1+7*v:8+7*v]=view_pair_cues(curves[i,v,:,classes[i,a]],curves[i,v,:,classes[i,b]])
 if not np.isfinite(out).all():raise ValueError('nonfinite features')
 return out
def design(raw,p,scale,arm):
 if arm not in ('C','V'):raise ValueError(arm)
 x=np.asarray(raw,float)/scale;x[:,:,1:]/=np.sqrt(14.)
 if arm=='C':x[:,:,1:]=0.
 uncertainty=4*np.asarray(p,float)*(1-np.asarray(p,float))
 return x*uncertainty[:,:,None]
def objective(beta,x,offset,targets,weights,penalty=1.0):
 scores=(offset+np.einsum('npd,d->np',x,beta))@BITS.T
 q=np.exp(scores-logsumexp(scores,axis=1,keepdims=True))
 loss=np.dot(weights,logsumexp(scores,axis=1)-scores[np.arange(len(scores)),targets])
 pair_error=q@BITS-BITS[targets]
 grad=np.einsum('n,np,npd->d',weights,pair_error,x)+penalty*beta
 return float(loss+.5*penalty*np.dot(beta,beta)),grad
def fit(raw,p,prior_logits,answers,repeat_groups,arm):
 if len(raw)!=len(answers):raise ValueError('row mismatch')
 scale=np.maximum(np.sqrt(np.mean(np.asarray(raw,float)**2,axis=(0,1))),.05);scale[0]=1.
 x=design(raw,p,scale,arm);offset=logit(p)+prior_logits
 target=np.asarray([np.flatnonzero(ANSWERS==a).item() for a in answers],int)
 _,inverse,counts=np.unique(np.asarray(repeat_groups,str),return_inverse=True,return_counts=True)
 weights=1/counts[inverse].astype(float);beta=np.zeros(raw.shape[-1],float)
 initial_loss,_=objective(beta,x,offset,target,weights)
 result=minimize(objective,beta,args=(x,offset,target,weights),jac=True,method='L-BFGS-B',
  options={'maxiter':300,'ftol':1e-12,'gtol':1e-7})
 if not np.isfinite(result.x).all() or result.fun>initial_loss+1e-8:raise RuntimeError('invalid optimizer outcome')
 return {'beta':result.x,'scale':scale,'arm':np.asarray(arm)}, {
  'success':bool(result.success),'message':str(result.message),'iterations':int(result.nit),
  'function_evaluations':int(result.nfev),'initial_loss':initial_loss,'final_loss':float(result.fun),
  'gradient_norm':float(np.linalg.norm(result.jac)),'effective_scene_weight':float(weights.sum()),
  'regularization':1.,'zero_init':True,'visual_features_enabled':arm=='V'}
def predict(model,raw,probability):
 delta=np.einsum('npd,d->np',design(raw,probability,model['scale'],str(model['arm'])),model['beta'])
 return expit(logit(probability)+delta),delta
