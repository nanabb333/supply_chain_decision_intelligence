import argparse
import json
from .fixture import generate
from .pipeline import run


def main():
    parser=argparse.ArgumentParser(description='Week 1 M5 pipeline; no forecasting or synthetic observed claims')
    sub=parser.add_subparsers(dest='command',required=True)
    fixture=sub.add_parser('fixture',help='Generate explicitly synthetic test inputs')
    fixture.add_argument('--output',default='data/fixture')
    ingest=sub.add_parser('ingest')
    ingest.add_argument('--raw',default='data/raw')
    ingest.add_argument('--output',default='data/processed')
    ingest.add_argument('--series',type=int,default=60,help='30–100, or 0 for all')
    ingest.add_argument('--seed',type=int,default=42)
    ingest.add_argument('--synthetic',action='store_true')
    args=parser.parse_args()
    if args.command=='fixture': generate(args.output)
    else: print(json.dumps(run(args.raw,args.output,args.series,args.seed,args.synthetic),indent=2))

if __name__=='__main__': main()
