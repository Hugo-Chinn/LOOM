#!/usr/bin/env python3
"""Create a tiny valid blocked TencentGR fixture for end-to-end CI."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]/'data'/'synthetic'/'TencentGR_10M_Action'
USER_SCALAR=['103','104','105','109']; USER_LIST=['106','107','108','110']
ITEM=['100','101','102','112','114','115','116','117','118','119','120','121','122']
action_vocab={'click':1,'click|conversion':2,'exposure':3}
rng=np.random.default_rng(20260922)
for split,n in [('train',256),('valid',128),('test',128)]:
    for kind in ['data','user_info','item_info']:(ROOT/split/kind).mkdir(parents=True,exist_ok=True)
    users=np.arange(1,33,dtype=np.int32); items=np.arange(1,129,dtype=np.int32)
    user=pd.DataFrame({'user_index':np.arange(33,dtype=np.int32),
                       'full_item_seq':[[]]+[[int((u+j)%128+1) for j in range(20)] for u in users],
                       'full_action_seq':[[]]+[[3 if j%3 else 1 for j in range(20)] for _ in users]})
    item={'item_index':np.arange(129,dtype=np.int32),'item_id':np.arange(129,dtype=np.int32)}
    for i,col in enumerate(ITEM):item[col]=(np.arange(129)%(i+3)+1).astype(np.int32)
    data={'user_index':rng.integers(1,33,n,dtype=np.int32),'item_index':rng.integers(1,129,n,dtype=np.int32),
          'seq_len':rng.integers(1,21,n,dtype=np.int32),'user_id':rng.integers(1,100,n,dtype=np.int32),
          'day_of_week':rng.integers(1,8,n,dtype=np.int32),'is_weekend':rng.integers(1,3,n,dtype=np.int32),
          'hour':rng.integers(1,25,n,dtype=np.int32)}
    for i,col in enumerate(USER_SCALAR):data[col]=rng.integers(1,i+4,n,dtype=np.int32)
    for i,col in enumerate(USER_LIST):data[col]=[list(rng.integers(1,i+4,5,dtype=np.int32)) for _ in range(n)]
    click=(rng.random(n)<.25).astype(np.float32); conv=((rng.random(n)<.15)&(click==1)).astype(np.float32)
    data['is_click']=click;data['is_conversion']=conv
    pd.DataFrame(data).to_parquet(ROOT/split/'data/part-00000.parquet',index=False)
    user.to_parquet(ROOT/split/'user_info/part-00000.parquet',index=False)
    pd.DataFrame(item).to_parquet(ROOT/split/'item_info/part-00000.parquet',index=False)
meta={'action_vocab':action_vocab,'feature_size':{**{x:10 for x in USER_SCALAR+USER_LIST+ITEM},
      'day_of_week':8,'is_weekend':3,'hour':25},'num_users':100,'num_items':129}
(ROOT/'meta_data.json').write_text(json.dumps(meta,indent=2)+'\n')
print(ROOT)
