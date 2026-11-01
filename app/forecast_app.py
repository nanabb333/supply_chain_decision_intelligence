"""E1 standalone entry point: accepted dashboards remain unchanged."""
from pathlib import Path
import sys
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
from supply_chain.forecast_dashboard import render
render()
