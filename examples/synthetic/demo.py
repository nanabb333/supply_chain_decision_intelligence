"""Fully synthetic public demonstration; never loads M5 data or trains models."""
from pathlib import Path
import argparse, json
import numpy as np
from supply_chain.fixture import generate
from supply_chain.pipeline import run
from supply_chain.baselines import score, training_stats
from supply_chain.inventory_engine import SyntheticFixture, simulate_synthetic
from supply_chain.inventory_protocol import Scenario
from supply_chain.inventory_reconcile import reconcile

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='outputs/synthetic-demo');a=p.parse_args()
    out=Path(a.output)
    if out.exists():raise FileExistsError('Use a fresh output directory')
    out.mkdir(parents=True)
    generate(out/'raw')
    run(out/'raw',out/'processed',synthetic=True)
    history=np.random.default_rng(42).poisson(2,(3,365)).astype(float)
    target=np.random.default_rng(43).poisson(2,(3,28)).astype(float)
    forecast=np.repeat(history[:,-28:].mean(axis=1)[:,None],28,axis=1)
    metrics=score(target,forecast,training_stats(history))
    metrics.to_csv(out/'SYNTHETIC_forecast_metrics.csv',index=False)
    fixture=SyntheticFixture('SYNTHETIC_PUBLIC_DEMO',((2.,)*28,(2.5,)*28,(3.,)*28),(3,0,7,1,8,0,0)*3)
    scenario=Scenario();results=simulate_synthetic(fixture,scenario)
    for result in results:reconcile(result,fixture,scenario)
    (out/'SYNTHETIC_inventory.json').write_text(json.dumps(results,indent=2))
    (out/'README.txt').write_text('All observations and forecast vectors are artificial. MA28/M1/M3 are policy interface labels here, not empirically fitted models. Costs are hypothetical. No M5 results reproduced.\n')
    print('Synthetic pipeline, forecast scoring and inventory reconciliation completed.')
if __name__=='__main__':main()
