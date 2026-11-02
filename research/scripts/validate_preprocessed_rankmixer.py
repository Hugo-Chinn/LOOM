#!/usr/bin/env python3
"""One-real-batch forward/backward validation for a local blocked dataset."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from unirank.features import FeatureMap
from unirank.preprocess import FeatureProcessor, build_dataset
from unirank.pytorch.dataloaders import RankDataLoader
from unirank.utils import load_config
import model_zoo

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--expid',required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    params=load_config(str(ROOT/'config_local'),args.expid)
    params.update(gpu=0,distributed=False,rank=0,local_rank=0,world_size=1,
                  enable_bf16=True,num_workers=0,enable_torch_compile=False)
    processor=FeatureProcessor(**params)
    params['train_data'],params['valid_data'],params['test_data']=build_dataset(processor,**params)
    data_dir=Path(params['data_root'])/params['dataset_id']
    fmap=FeatureMap(params['dataset_id'],str(data_dir))
    fmap.load(str(data_dir/'feature_map.json'),params)
    model=getattr(model_zoo,params['model'])(fmap,**params).cuda()
    loader,_=RankDataLoader(fmap,stage='train',**params).make_iterator()
    batch=next(iter(loader))
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast('cuda',dtype=torch.bfloat16): output=model(batch)
    labels=model.get_labels(batch)
    loss=model.compute_loss({k:v.float() for k,v in output.items()},labels)
    loss.backward()
    result={'dataset':params['dataset_id'],'expid':args.expid,
      'tasks':list(fmap.labels),'batch_size':int(labels[0].shape[0]),
      'output_shapes':{k:list(v.shape) for k,v in output.items()},
      'loss':float(loss.detach()),'finite_loss':bool(torch.isfinite(loss)),
      'finite_gradients':all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None),
      'total_parameters':sum(p.numel() for p in model.parameters()),
      'embedding_parameters':sum(p.numel() for p in model.embedding_layer.parameters()),
      'peak_gpu_memory_bytes':torch.cuda.max_memory_allocated(),
      'history_mask_shape':list(batch[2].shape),'history_nonzero':int(batch[2].sum())}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__': main()
