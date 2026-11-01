"""Week3-A pure protocol guardrails. No inventory simulation or ordering engine."""
from dataclasses import dataclass,asdict
from pathlib import Path
import math,hashlib,json
@dataclass(frozen=True)
class Scenario:
 name:str='primary'
 lead_days:int=3
 review_days:int=7
 episode_days:int=21
 initial_multiplier:float=1.0
 initial_pipeline_units:float=0.0
 buffer_days:float=3.0
 holding_per_unit_day:float=1.0
 shortage_per_unit:float=10.0
 ordering_per_order:float=0.0
 logic:str='lost_sales'
 cost_unit:str='hypothetical cost units'
 def validate(self):
  for name in ['lead_days','review_days','episode_days']:
   value=getattr(self,name)
   if type(value)!=int or value<1:raise ValueError(name)
  for name in ['initial_multiplier','initial_pipeline_units','buffer_days','holding_per_unit_day','shortage_per_unit','ordering_per_order']:
   value=getattr(self,name)
   if isinstance(value,bool) or not isinstance(value,(float,int)) or not math.isfinite(value) or value<0:raise ValueError(name)
  if self.initial_pipeline_units!=0:raise ValueError('Nonzero pipeline requires explicit due-date ledger; unavailable in A scaffold')
  if self.logic!='lost_sales':raise ValueError('Backorder extension not accepted in this protocol')
  if self.cost_unit!='hypothetical cost units':raise ValueError('No observed monetary costs')
  for t in range(1,self.episode_days+1,self.review_days):protection_window(t,self.lead_days,self.review_days)
  return self

def protection_window(day,lead,review,horizon=28):
 for x in [day,lead,review,horizon]:
  if type(x)!=int or x<1:raise ValueError('Positive integer day indices required')
 end=day+lead+review-1
 if end>horizon:raise ValueError('Forecast coverage exhausted; extrapolation forbidden')
 return tuple(range(day,end+1))

def arrival_day(order_day,lead):
 if type(order_day)!=int or type(lead)!=int or order_day<1 or lead<1:raise ValueError('No same-day arrival')
 return order_day+lead

def information_gate(origin,decision_day,forecast_origin,history_max):
 if any(type(x)!=int for x in [origin,decision_day,forecast_origin,history_max]):raise ValueError('Integer time indices required')
 if decision_day<=origin or forecast_origin>origin or forecast_origin>=decision_day or history_max>=decision_day:raise ValueError('Information is not available before demand')
 if forecast_origin!=origin:raise ValueError('Episode uses its declared fixed origin only')
 return True

def initial_inventory(mean28,scenario):
 scenario.validate()
 if mean28 is None or isinstance(mean28,bool) or not math.isfinite(mean28) or mean28<0:raise ValueError('Verified past mean required; missing is not zero')
 return math.ceil(scenario.initial_multiplier*(scenario.lead_days+scenario.review_days+scenario.buffer_days)*mean28)

def require_inputs(mapping):
 required={'sales_proxy','series_keys','saved_forecast','issue_convention','scenario'}
 if required-mapping.keys() or any(mapping[k] is None for k in required):raise ValueError('Required inputs absent')
 if mapping['issue_convention']!='end_of_origin_day_assumed':raise ValueError('Issue timing must be explicit')
 if not isinstance(mapping['scenario'],Scenario):raise ValueError('Explicit scenario required')
 mapping['scenario'].validate();return True

def read_protected(root,relative,manifest):
 """Return immutable bytes only after verifying an allowlisted SHA-256."""
 root=Path(root).resolve();p=(root/relative).resolve()
 if not p.is_relative_to(root) or relative not in manifest:raise ValueError('Outside protected allowlist')
 data=p.read_bytes()
 if hashlib.sha256(data).hexdigest()!=manifest[relative]:raise ValueError('Protected artifact mismatch')
 return data

def reserve_output(path,protected_roots):
 p=Path(path).resolve()
 if p.exists() or any(p.is_relative_to(Path(r).resolve()) for r in protected_roots):raise ValueError('Refusing existing/protected output')
 return p

def validate_config(config):
 if config.get('simulation_enabled') is not False:raise ValueError('Checkpoint A cannot enable simulation')
 if config.get('final_origin_primary') is not False or config.get('development_origins')!=[1801,1829,1857]:raise ValueError('Final origin excluded from primary design')
 if config.get('models')!=['MA28','M1','M3'] or config.get('forecast_horizon')!=28:raise ValueError('Saved forecast design must remain explicit')
 target=config.get('service_target_assumption')
 if isinstance(target,bool) or not isinstance(target,(int,float)) or not math.isfinite(target) or not 0<target<=1:raise ValueError('Invalid hypothetical service reference')
 scenarios=config.get('scenarios',[])
 if not scenarios or len({s['name'] for s in scenarios})!=len(scenarios):raise ValueError('Missing or duplicate scenarios')
 for record in scenarios:
  scenario=Scenario(**record).validate()
  if scenario.episode_days!=config.get('episode_days'):raise ValueError('Unmatched episode horizon')
 return True
