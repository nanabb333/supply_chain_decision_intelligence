"""Week3 E1 versioned portfolio dashboard; empirical research frozen."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from supply_chain.portfolio_dashboard import render
render()
