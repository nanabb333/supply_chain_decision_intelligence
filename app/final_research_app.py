"""Week2 F publication entry; accepted E1 remains unchanged."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from supply_chain.publication_dashboard import render
render()
