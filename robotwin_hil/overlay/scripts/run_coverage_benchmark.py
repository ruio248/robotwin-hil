#!/usr/bin/env python3
"""Reproducible full-chunk benchmark: fit / offline / branch / report."""
import argparse
import importlib
from pathlib import Path
import traceback
from coverage_benchmark.common import load_config, identity, atomic_json, RunLock


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('fit','offline','branch','report'))
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--profile',choices=('new_power_3','ubuntu'),required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--gate-only',action='store_true',help='branch only: 3-state replay validation without formal schemes')
    p.add_argument('--validation-only',action='store_true',help='offline only: internal 30-episode diagnostic, not full offline result')
    args=p.parse_args(); config=load_config(args.config,args.profile)
    if args.gate_only and args.phase!='branch':p.error('--gate-only is only for branch')
    config['_gate_only']=args.gate_only
    if args.validation_only and args.phase!='offline':p.error('--validation-only is only for offline')
    config['_validation_only']=args.validation_only
    output=args.output_dir; output.mkdir(parents=True,exist_ok=True)
    with RunLock(output/'worker.pid'):
        identity(output,config,args.resume)
        from coverage_benchmark.common import sha256
        code={str(f.relative_to(Path(__file__).parent)):sha256(f) for f in
              (Path(__file__).parent/'coverage_benchmark').glob('*.py')}
        atomic_json(output/'code_provenance.json',code)
        atomic_json(output/'status.json',{'phase':args.phase,'state':'running'})
        try:
            importlib.import_module('coverage_benchmark.'+args.phase).run(config,output,args.resume)
        except BaseException as exc:
            atomic_json(output/'status.json',{'phase':args.phase,'state':'failed',
                'error':repr(exc),'traceback':traceback.format_exc()})
            raise
        atomic_json(output/'status.json',{'phase':args.phase,'state':'complete'})


if __name__=='__main__': main()
