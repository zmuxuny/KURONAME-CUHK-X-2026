"""D1 matched coarse/fine temporal heads on the paired server-runtime bank."""
from pathlib import Path
import json
import numpy as np

from cuhkx.action_mil import positive_weights
from cuhkx.action_window_head import CHANNELS, VIEWS, WindowPartialActionClassifier
from cuhkx.action_joint_harn_head import HarnWindowBank
from cuhkx.harn import action_class
from cuhkx.metadata import subject_id
from cuhkx.sequence_skeleton_nested import array_digest
from cuhkx.server_temporal_head import pool_per_view, fuse_views, partial_loss
from cuhkx.validation import sha256_file

MODALITIES=("Depth_Color","IR")

def read_d1_bank(directory, split, mode):
    directory=Path(directory); manifest=json.loads((directory/'manifest.json').read_text())
    if manifest['status']!='complete' or manifest['labels_used'] or manifest['competition_fits']!=0:
        raise ValueError('D1 paired bank is not a complete label-free representation')
    for entry in manifest['files']:
        p=directory/entry['path']
        if p.stat().st_size!=entry['bytes'] or sha256_file(p)!=entry['sha256']: raise ValueError('D1 bank changed')
    with np.load(directory/'metadata.npz',allow_pickle=False) as a: meta={k:a[k] for k in a.files}
    keep=np.flatnonzero(meta['split'].astype(str)==split)
    units=meta['units'][keep].astype(str); mods=meta['modalities'][keep].astype(str); avail=meta['available'][keep]
    if mode=='coarse': raw=np.load(directory/'coarse.npy',mmap_mode='r',allow_pickle=False)[keep]; bins=16
    elif mode=='fine': raw=np.load(directory/'fine.npy',mmap_mode='r',allow_pickle=False)[keep].reshape(len(keep),128,CHANNELS); bins=128
    else: raise ValueError('mode must be coarse or fine')
    lookup={(u,m):i for i,(u,m) in enumerate(zip(units,mods,strict=True))}
    ordered=sorted(set(units)); values=np.zeros((len(ordered),bins,VIEWS,CHANNELS),np.float32); mask=np.zeros((len(ordered),bins,VIEWS),bool)
    for i,u in enumerate(ordered):
        for v,m in enumerate(MODALITIES):
            r=lookup[(u,m)]; values[i,:,v]=raw[r]; mask[i,:,v]=bool(avail[r])
    if not mask.any((1,2)).all() or not np.isfinite(values).all(): raise ValueError('invalid D1 unit matrix')
    matrix=np.concatenate([values.reshape(len(ordered),-1),mask.reshape(len(ordered),-1)],axis=1)
    return {'X':matrix,'units':np.asarray(ordered),'index':{u:i for i,u in enumerate(ordered)},
            'feature_names':np.asarray([f'd1_{mode}_paired_server_v1']), 'bins':bins,'mode':mode,
            'source_manifest_sha256':sha256_file(directory/'manifest.json')}

def unpack_d1(matrix,bins):
    x=np.asarray(matrix,dtype=np.float32); nfeat=bins*VIEWS*CHANNELS; nmask=bins*VIEWS
    if x.ndim!=2 or x.shape[1]!=nfeat+nmask or not np.isfinite(x).all(): raise ValueError('bad D1 matrix')
    f=x[:,:nfeat].reshape(-1,bins,VIEWS,CHANNELS); raw=x[:,nfeat:]
    if not np.isin(raw,[0,1]).all(): raise ValueError('D1 mask must be binary')
    m=raw.reshape(-1,bins,VIEWS).astype(bool)
    if not m.any((1,2)).all(): raise ValueError('D1 recording lacks all views')
    return f,m

class D1HarnBank(HarnWindowBank):
    def __init__(self,directory):
        directory=Path(directory); self.manifest_path=str(directory/'manifest.json'); self.manifest_sha256=sha256_file(Path(self.manifest_path))
        manifest=json.loads(Path(self.manifest_path).read_text())
        if manifest['kind']!='cuhkx_server_d1_harn_whole_v1' or manifest['labels_used']: raise ValueError('wrong D1 HARn bank')
        for entry in manifest['files']:
            p=directory/entry['path']
            if p.stat().st_size!=entry['bytes'] or sha256_file(p)!=entry['sha256']: raise ValueError('D1 HARn bank changed')
        with np.load(directory/'metadata.npz',allow_pickle=False) as a: meta={k:a[k] for k in a.files}
        raw=np.load(directory/'features.npy',allow_pickle=False)
        keys=list(zip(meta['units'].astype(str),meta['modalities'].astype(str),strict=True)); lookup={k:i for i,k in enumerate(keys)}
        self.units=np.asarray(sorted({k[0] for k in keys})); self.features=np.zeros((len(self.units),1,2,768),np.float32); self.mask=np.zeros((len(self.units),1,2),bool)
        for i,u in enumerate(self.units):
            rows=[lookup[(u,m)] for m in MODALITIES]; self.features[i,0]=raw[rows,0]; self.mask[i,0]=meta['available'][rows]
            if self.mask[i].all() and not meta['alignment_ok'][rows].all(): self.mask[i,0,0]=False
        self.subjects=np.asarray([subject_id(u) for u in self.units])

class D1TemporalClassifier(WindowPartialActionClassifier):
    def __init__(self,config,seed,auxiliary,bins):
        super().__init__(config,seed); self.auxiliary=auxiliary; self.bins=int(bins)

    def fit(self,matrix,targets):
        import torch
        features,mask=unpack_d1(matrix,self.bins)
        auxiliary,auxiliary_mask,auxiliary_labels,self.auxiliary_scope=self.auxiliary; del self.auxiliary
        weights,self.supported=positive_weights(targets); self.fitting_content_sha256=array_digest(np.asarray(matrix),targets)
        self.training_targets=np.asarray(targets).copy(); self.n_features_in_,self.n_outputs_=matrix.shape[1],40
        self.mean=np.zeros((VIEWS,CHANNELS),np.float32); self.scale=np.ones_like(self.mean)
        for view in range(VIEWS):
            visible=features[:,:,view][mask[:,:,view]]
            if len(visible): self.mean[view]=visible.mean(0); self.scale[view]=np.maximum(visible.std(0),self.config['std_floor'])
        device=self.config['device']; torch.manual_seed(self.seed)
        base=torch.nn.Sequential(torch.nn.Linear(CHANNELS,self.config['hidden']),torch.nn.GELU(),torch.nn.Dropout(self.config['dropout']),torch.nn.Linear(self.config['hidden'],40)).to(device)
        optimizer=torch.optim.AdamW(base.parameters(),lr=self.config['lr'],weight_decay=self.config['weight_decay'])
        steps=self.config['epochs']*((len(features)+self.config['batch_size']-1)//self.config['batch_size']); scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=steps)
        torch.manual_seed(self.seed+424242); random=np.random.default_rng(self.seed); aux_random=np.random.default_rng(self.seed+9973)
        data=torch.from_numpy(self._standardized(features,mask)).to(device); visible=torch.from_numpy(mask).to(device); labels=torch.as_tensor(targets,device=device,dtype=torch.float32); pw=torch.from_numpy(weights).to(device)
        aux_data=torch.from_numpy(self._standardized(auxiliary,auxiliary_mask)).to(device); aux_mask=torch.from_numpy(auxiliary_mask[:,0]).to(device); aux_labels=torch.from_numpy(auxiliary_labels).to(device)
        counts=np.bincount(auxiliary_labels,minlength=40); sampling=1/counts[auxiliary_labels].astype(np.float64); sampling/=sampling.sum()
        import time; started=time.perf_counter(); self.history=[]
        for epoch in range(self.config['epochs']):
            main_total=aux_total=0.; seen=0; order=random.permutation(len(features))
            for start in range(0,len(order),self.config['batch_size']):
                rows=torch.as_tensor(order[start:start+self.config['batch_size']],device=device); extra=torch.as_tensor(aux_random.choice(len(auxiliary),size=len(rows),replace=True,p=sampling),device=device)
                optimizer.zero_grad(set_to_none=True); hidden=torch.nn.functional.gelu(base[0](data[rows])); logits=base[3](base[2](hidden))
                views,present=pool_per_view(logits,visible[rows],self.config['temperature']); pooled=fuse_views(views,present)
                main=partial_loss(pooled,labels[rows],pw)
                ah=torch.nn.functional.gelu(base[0](aux_data[extra])); al=base[3](base[2](ah))[:,0]
                ap=(al*aux_mask[extra,:,None]).sum(1)/aux_mask[extra].sum(1)[:,None]; aux=torch.nn.functional.cross_entropy(ap,aux_labels[extra])
                loss=main+self.config['auxiliary_weight']*aux
                if not torch.isfinite(loss): raise ValueError('nonfinite D1 loss')
                loss.backward(); torch.nn.utils.clip_grad_norm_(base.parameters(),self.config['gradient_clip']); optimizer.step(); scheduler.step()
                main_total+=float(main.detach())*len(rows); aux_total+=float(aux.detach())*len(rows); seen+=len(rows)
            self.history.append({'epoch':epoch+1,'mean_training_loss':main_total/seen,'auxiliary_ce':aux_total/seen})
        self.state={k:v.detach().cpu().numpy().copy() for k,v in base.state_dict().items()}; self.positive_weight=weights; self.fit_seconds=time.perf_counter()-started
        if any(not np.isfinite(v).all() for v in self.state.values()): raise ValueError('nonfinite D1 state')
        return self

    def predict(self,matrix):
        import torch
        features,mask=unpack_d1(matrix,self.bins); data=torch.from_numpy(self._standardized(features,mask)); visible=torch.from_numpy(mask)
        with torch.inference_mode():
            h=torch.nn.functional.gelu(torch.nn.functional.linear(data,torch.from_numpy(self.state['0.weight']),torch.from_numpy(self.state['0.bias'])))
            logits=torch.nn.functional.linear(h,torch.from_numpy(self.state['3.weight']),torch.from_numpy(self.state['3.bias']))
            views,present=pool_per_view(logits,visible,self.config['temperature']); pooled=fuse_views(views,present)
            p=(pooled-torch.from_numpy(np.log(self.positive_weight))).sigmoid().numpy()
        p[:,~self.supported]=.5; return p
