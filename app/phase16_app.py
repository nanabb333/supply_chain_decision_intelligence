"""Checkpoint E dashboard extension; accepted Phase 1.5 entry point stays intact."""
from pathlib import Path
import sys,runpy
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
import streamlit as st
section=st.sidebar.radio('Section',['Classification Robustness & Diagnostic Quality','Sales performance'],key='section')
if section=='Sales performance':
    runpy.run_path(str(R/'app/streamlit_app.py'),run_name='__main__')
else:
    from supply_chain.diagnostic_dashboard import render
    render(R)
