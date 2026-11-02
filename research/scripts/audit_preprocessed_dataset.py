#!/usr/bin/env python3
"""Audit a downloaded UniRank blocked dataset using local files and metadata."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pyarrow.parquet as pq


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('dataset_dir', type=Path)
    ap.add_argument('--output', type=Path)
    args=ap.parse_args()
    root=args.dataset_dir.resolve()
    audit=json.loads((root/'download_audit.json').read_text())
    manifest=json.loads((root/'block_manifest.json').read_text())
    meta=json.loads((root/'meta_data.json').read_text())
    result={'dataset':root.name,'download_complete':bool(audit['complete']),
            'revision':audit['revision'],'splits':{},'errors':[]}
    for split in ('train','valid','test'):
        blocks=manifest[split]['blocks']
        expected_rows=sum(int(x['rows']) for x in blocks)
        files={kind:sorted((root/split/kind).glob('part-*.parquet'))
               for kind in ('data','user_info','item_info')}
        ids={kind:[p.stem for p in paths] for kind,paths in files.items()}
        paired=ids['data']==ids['user_info']==ids['item_info']
        actual_rows=sum(pq.ParquetFile(p).metadata.num_rows for p in files['data'])
        if len(files['data'])!=len(blocks): result['errors'].append(f'{split}: block count mismatch')
        if actual_rows!=expected_rows: result['errors'].append(f'{split}: row count mismatch')
        if not paired: result['errors'].append(f'{split}: unpaired part ids')
        result['splits'][split]={'blocks':len(files['data']),'rows':actual_rows,
                                'manifest_rows':expected_rows,'paired':paired}
    declared=meta.get('sample_size',{})
    result['declared_sample_size']=declared
    for split in ('train','valid','test'):
        if split in declared and int(declared[split])!=result['splits'][split]['rows']:
            result['errors'].append(f'{split}: metadata row count mismatch')
    result['complete']=result['download_complete'] and not result['errors']
    out=args.output or root/'local_audit.json'
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['complete'] else 1)
if __name__=='__main__': main()
