#!/usr/bin/env python3
"""Reproduce the selected calibrated-slot replacement, retaining all old gates."""
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np
from scripts import hau_action_checkpoint as hau
from scripts import hau_action_acquisition as acquisition

def main():
 p=argparse.ArgumentParser(add_help=False);p.add_argument('--source-presence',type=Path,required=True)
 a,rest=p.parse_known_args()
 with np.load(a.source_presence) as z:replacement=dict(zip(z['units'].astype(str),z['probabilities']))
 original=hau.broad.decode_frame
 def decode(qa,sources,*args,**kwargs):
  keys=set(sources['calibrated'])
  if not keys<=set(replacement):raise ValueError('Source presence does not cover current HAU clips')
  return original(qa,{**sources,'calibrated':{u:replacement[u] for u in keys}},*args,**kwargs)
 hau.broad.decode_frame=decode
 sys.argv=[sys.argv[0],*rest]
 return acquisition.main()
if __name__=='__main__':main()
