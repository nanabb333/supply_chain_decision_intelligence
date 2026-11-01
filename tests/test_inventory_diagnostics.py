import numpy as np
import pandas as pd
import pytest
from supply_chain.inventory_diagnostics import compare_numeric,compare_category,attach_sample

def test_identical():
 r=compare_numeric([0,1,2],[0,1,2]);assert r['ks_distance']==0 and r['standardized_mean_difference']==0

def test_ks_hand():
 r=compare_numeric([0,0],[0,1,2,3]);assert r['ks_distance']==.75

def test_missing_and_constant():
 r=compare_numeric([1,np.nan],[1,1]);assert r['sample_missing']==1 and r['standardized_mean_difference'] is None

def test_all_missing():assert compare_numeric([np.nan],[0])['ks_distance'] is None

def test_infinity_rejected():
 with pytest.raises(ValueError):compare_numeric([np.inf],[0])

def test_category_hand():
 rows,tv=compare_category(['X','X'],['X','Y']);assert tv==.5 and sum(r['sample_count'] for r in rows)==2

def test_category_undefined():
 rows,tv=compare_category([None],['X']);assert tv==1 and 'UNDEFINED' in {r['level'] for r in rows}

def test_empty_rejected():
 with pytest.raises(ValueError):compare_category([],['X'])

def test_membership():
 p=pd.DataFrame({'item_id':['a','b'],'store_id':['s','s']});assert attach_sample(p,p.iloc[:1]).selected.tolist()==[True,False]

def test_duplicate_rejected():
 p=pd.DataFrame({'item_id':['a','a'],'store_id':['s','s']})
 with pytest.raises(ValueError):attach_sample(p,p.iloc[:1])

def test_missing_id_rejected():
 p=pd.DataFrame({'item_id':['a'],'store_id':['s']});q=pd.DataFrame({'item_id':['b'],'store_id':['s']})
 with pytest.raises(ValueError):attach_sample(p,q)
