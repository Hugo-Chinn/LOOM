#!/usr/bin/env python3
"""Extract final TAAC RankMixer metrics and compare with repository benchmark."""
from __future__ import annotations
import argparse, json, re
from datetime import datetime
from pathlib import Path

LINE=re.compile(r'^(?P<ts>\d{4}-\d\d-\d\d [\d:,]+).*\[Task: (?P<task>[^]]+)\]\[Metrics\] (?P<metrics>.*)$')
PAIR=re.compile(r'(AUC|logloss): ([0-9.]+)')
PARAM=re.compile(r'Total number of parameters: (\d+)')

def parse(path: Path):
    text=path.read_text(errors='replace')
    test=text.split('******** Test evaluation ********')[-1] if '******** Test evaluation ********' in text else ''
    metrics={}
    for line in test.splitlines():
        m=LINE.search(line)
        if m: metrics[m['task']]={k:float(v) for k,v in PAIR.findall(m['metrics'])}
    pm=PARAM.search(text)
    ts=[]
    for line in text.splitlines():
        if len(line)>=23:
            try: ts.append(datetime.strptime(line[:23],'%Y-%m-%d %H:%M:%S,%f'))
            except ValueError: pass
    return {'complete':'Training finished.' in text and bool(metrics),
            'parameters':int(pm.group(1)) if pm else None,'test':metrics,
            'runtime_seconds':(ts[-1]-ts[0]).total_seconds() if len(ts)>=2 else None}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--local',type=Path,required=True); ap.add_argument('--official',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); args=ap.parse_args()
    local=parse(args.local); official=parse(args.official)
    out={'local':local,'official':official,'deltas':{},'status':'complete' if local['complete'] else 'running'}
    for task, vals in local['test'].items():
        if task in official['test']:
            out['deltas'][task]={k:vals[k]-official['test'][task][k] for k in vals if k in official['test'][task]}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
    raise SystemExit(0 if local['complete'] else 2)
if __name__=='__main__': main()
