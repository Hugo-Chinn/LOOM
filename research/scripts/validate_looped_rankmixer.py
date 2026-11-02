#!/usr/bin/env python3
"""Sanity-check all first-stage Taobao RankMixer variants on one real batch."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from unirank.features import FeatureMap
from unirank.preprocess import FeatureProcessor,build_dataset
from unirank.pytorch.dataloaders import RankDataLoader
from unirank.utils import load_config
import model_zoo

DEFAULT=[
'RankMixer_Taobao_P1_Parallel','RankMixer_Taobao_P1_Sequential',
'RankMixer_Taobao_P1_Looped','RankMixer_Taobao_P1_Looped_NoBehavior',
'RankMixer_Taobao_P1_Looped_SingleToken','RankMixer_Taobao_P1_Looped_MultiToken',
'RankMixer_Taobao_P1_Looped_DeepSupervision','RankMixer_Taobao_P1_Looped_WholeEncoder']

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--expids',nargs='*',default=DEFAULT); ap.add_argument('--output',type=Path,default=ROOT.parent/'reports/unirank_taobao_loop_sanity.json'); args=ap.parse_args()
 results=[]
 for expid in args.expids:
  p=load_config(str(ROOT/'config_local'),expid); p.update(gpu=0,distributed=False,rank=0,local_rank=0,world_size=1,enable_bf16=True,num_workers=0,enable_torch_compile=False,batch_size=128,enable_loop_diagnostics=True)
  fp=FeatureProcessor(**p); p['train_data'],p['valid_data'],p['test_data']=build_dataset(fp,**p); dd=Path(p['data_root'])/p['dataset_id']; fm=FeatureMap(p['dataset_id'],str(dd)); fm.load(str(dd/'feature_map.json'),p)
  model=model_zoo.RankMixer(fm,**p).cuda(); loader,_=RankDataLoader(fm,stage='train',**p).make_iterator(); batch=next(iter(loader)); torch.cuda.reset_peak_memory_stats()
  with torch.autocast('cuda',dtype=torch.bfloat16): out=model(batch)
  labels=model.get_labels(batch); loss=model.compute_loss({k:v.float() for k,v in out.items()},labels); loss.backward()
  loop_grads=[]
  if model.loop_block is not None: loop_grads=[x.grad is not None and bool(torch.isfinite(x.grad).all()) for x in model.loop_block.parameters()]
  registered_loop_groups=1 if model.loop_block is not None else (len(model.loop_blocks) if model.loop_blocks is not None else 0)
  results.append({'expid':expid,'model_type':model.model_type,'loop_scope':model.loop_scope,'supervision_mode':model.supervision_mode,'task_token_mode':model.task_token_mode,'use_behavior_embedding':model.use_behavior_embedding,'parameters':sum(x.numel() for x in model.parameters()),'dense_parameters':sum(x.numel() for x in model._dense_params),'registered_loop_groups':registered_loop_groups,'shared_loop_verified':model.model_type!='looped' or registered_loop_groups==1,'loss':float(loss.detach()),'finite_gradients':all(bool(torch.isfinite(x.grad).all()) for x in model.parameters() if x.grad is not None),'loop_gradients':all(loop_grads) if loop_grads else None,'diagnostics':getattr(model,'last_loop_diagnostics',None),'peak_gpu_memory_bytes':torch.cuda.max_memory_allocated()})
  del model,batch,out,loss; torch.cuda.empty_cache()
 args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(results,indent=2)+'\n'); print(json.dumps(results,indent=2))
if __name__=='__main__':main()
