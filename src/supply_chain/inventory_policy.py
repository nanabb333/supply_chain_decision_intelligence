"""Week 3 B: pure policy contracts/calculations. No episode execution engine."""
from dataclasses import dataclass
import math
import re
from typing import Literal, Mapping
from .inventory_protocol import Scenario, initial_inventory, protection_window, information_gate

MODELS = ('MA28', 'M1', 'M3')
ORIGINS = (1801, 1829, 1857)


def quantity(value, name, *, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f'{name}: finite nonnegative quantity required')
    if integer and type(value) is not int:
        raise ValueError(f'{name}: integer required')
    return value


@dataclass(frozen=True)
class PolicyConfig:
    model: Literal['MA28', 'M1', 'M3']
    rule: str = 'periodic_order_up_to'
    rounding: str = 'ceil_positive_gap'
    forecast_updates: bool = False

    def validate(self):
        if self.model not in MODELS or self.rule != 'periodic_order_up_to' or self.rounding != 'ceil_positive_gap' or self.forecast_updates is not False:
            raise ValueError('Only frozen matched MA28/M1/M3 policy definitions are allowed')
        return self


@dataclass(frozen=True)
class ForecastInput:
    item_id: str
    store_id: str
    origin: int
    model: str
    values: tuple[float, ...]
    source: str
    source_sha256: str
    history_max_day: int
    issue_convention: str = 'end_of_origin_day_assumed'

    def validate(self, manifest: Mapping[str, str]):
        if not all(isinstance(x, str) and x.strip() for x in (self.item_id, self.store_id)):
            raise ValueError('Explicit item/store keys required')
        if type(self.origin) is not int or self.origin not in ORIGINS or self.model not in MODELS:
            raise ValueError('Development fixed-origin inputs only')
        if type(self.values) is not tuple or len(self.values) != 28:
            raise ValueError('Exactly 28 immutable forecast horizons required')
        for value in self.values:
            quantity(value, 'forecast')
        if type(self.history_max_day) is not int or self.history_max_day != self.origin:
            raise ValueError('Forecast historical cutoff must equal fixed origin')
        if self.issue_convention != 'end_of_origin_day_assumed':
            raise ValueError('Unobserved issue time must use the explicit convention')
        if not isinstance(self.source_sha256, str) or not re.fullmatch('[0-9a-f]{64}', self.source_sha256) or manifest.get(self.source) != self.source_sha256:
            raise ValueError('Source must match protected manifest')
        if self.model == 'MA28':
            expected = rf'reports/week2/checkpoint-b-v1/full-v1/forecasts/{self.origin}-[0-9]+\.parquet'
            if len(set(self.values)) != 1:
                raise ValueError('MA28 must be the saved constant historical rate')
        else:
            expected = rf'reports/week2/checkpoint-c-v2/population-{self.origin}/{self.model}-c0-forecasts\.parquet'
        if not isinstance(self.source, str) or re.fullmatch(expected, self.source) is None:
            raise ValueError('Wrong model/origin/candidate source')
        return self


@dataclass(frozen=True)
class MatchedInputs:
    forecasts: tuple[ForecastInput, ...]

    def validate(self, manifest):
        if type(self.forecasts) is not tuple or len(self.forecasts) != 3 or {f.model for f in self.forecasts} != set(MODELS):
            raise ValueError('Exactly one input for each matched model required')
        for forecast in self.forecasts:
            forecast.validate(manifest)
        if len({(f.item_id, f.store_id, f.origin) for f in self.forecasts}) != 1:
            raise ValueError('Unmatched series/origin')
        return self

    def forecast(self, model):
        return next(f for f in self.forecasts if f.model == model)


@dataclass(frozen=True)
class OrderEvent:
    order_id: str
    placed_day: int
    due_day: int
    units: int

    def validate(self, scenario):
        if not isinstance(self.order_id, str) or not self.order_id:
            raise ValueError('Unique order identifier required')
        for name in ('placed_day', 'due_day', 'units'):
            quantity(getattr(self, name), name, integer=True)
        if not 1 <= self.placed_day <= scenario.episode_days or (self.placed_day - 1) % scenario.review_days:
            raise ValueError('Orders only on episode review days')
        if self.units == 0 or self.due_day != self.placed_day + scenario.lead_days:
            raise ValueError('Positive order with exact delayed arrival required')
        return self


@dataclass(frozen=True)
class InventoryState:
    """Start-of-day snapshot AFTER receipts, BEFORE ordering and current-day demand."""
    item_id: str
    store_id: str
    origin: int
    policy_model: str
    scenario: Scenario
    day: int
    on_hand: int
    pipeline: tuple[OrderEvent, ...] = ()
    backlog: int = 0
    observed_through: int = 0

    def validate(self):
        self.scenario.validate()
        if type(self.day) is not int or not 1 <= self.day <= self.scenario.episode_days:
            raise ValueError('Day outside episode')
        quantity(self.on_hand, 'on_hand', integer=True)
        quantity(self.backlog, 'backlog', integer=True)
        if self.backlog != 0 or type(self.pipeline) is not tuple:
            raise ValueError('Lost-sales state requires zero backlog and immutable pipeline')
        if self.policy_model not in MODELS or self.origin not in ORIGINS:
            raise ValueError('Invalid state identity')
        if type(self.observed_through) is not int or self.observed_through != self.origin + self.day - 1:
            raise ValueError('State must reflect all prior days and no current-day sales')
        ids = set()
        for event in self.pipeline:
            event.validate(self.scenario)
            if event.order_id in ids or event.due_day <= self.day or event.placed_day >= self.day:
                raise ValueError('Duplicate, overdue, unreceived or not-yet-placed pipeline event')
            ids.add(event.order_id)
        return self


def initialize(inputs, policy, scenario, manifest):
    inputs.validate(manifest); policy.validate(); scenario.validate()
    f = inputs.forecast(policy.model)
    mean28 = inputs.forecast('MA28').values[0]
    return InventoryState(f.item_id, f.store_id, f.origin, policy.model, scenario, 1,
                          initial_inventory(mean28, scenario), observed_through=f.origin).validate()


@dataclass(frozen=True)
class OrderCalculation:
    review: bool
    protection_horizons: tuple[int, ...]
    target: float | None
    inventory_position: int
    order_units: int
    due_day: int | None


def calculate_order(inputs, policy, state, manifest):
    """Calculate one decision; no demand argument, state mutation, queue or episode loop."""
    inputs.validate(manifest); policy.validate(); state.validate()
    f = inputs.forecast(policy.model)
    if (state.item_id, state.store_id, state.origin, state.policy_model) != (f.item_id, f.store_id, f.origin, policy.model):
        raise ValueError('State does not belong to this policy/series/episode')
    information_gate(f.origin, f.origin + state.day, f.origin, state.observed_through)
    ip = state.on_hand + sum(event.units for event in state.pipeline)
    if (state.day - 1) % state.scenario.review_days:
        return OrderCalculation(False, (), None, ip, 0, None)
    horizons = protection_window(state.day, state.scenario.lead_days, state.scenario.review_days)
    target = math.fsum(f.values[j-1] for j in horizons) + state.scenario.buffer_days * inputs.forecast('MA28').values[0]
    quantity(target, 'target')
    order = max(0, math.ceil(target - ip))
    return OrderCalculation(True, horizons, target, ip, order,
                            state.day + state.scenario.lead_days if order else None)


def validate_design(config, a_config):
    """Reject drift in A's matrix; no function here enables execution."""
    from .inventory_protocol import validate_config
    validate_config(a_config)
    if config.get('scenarios') != a_config['scenarios']:
        raise ValueError('A scenario definitions must be preserved exactly')
    if config.get('simulation_enabled') is not False or config.get('pilot_execution_authorized') is not False:
        raise ValueError('Checkpoint B does not authorize execution')
    if config.get('development_origins') != list(ORIGINS) or config.get('forecast_horizon') != 28 or config.get('episode_days') != 21:
        raise ValueError('Fixed-origin development design changed')
    if config.get('policies') != [dict(model=m, rule='periodic_order_up_to', rounding='ceil_positive_gap', forecast_updates=False) for m in MODELS]:
        raise ValueError('Matched policy configuration changed')
    if config.get('pilot') != dict(origin=1801, series=128, episode_days=21, scenarios=['primary'], selection='sha256(item_id|store_id), digest/item/store ascending'):
        raise ValueError('Pilot scope differs from declared minimal design')
    for record in config['scenarios']:
        Scenario(**record).validate()
    if config.get('assumption_status') not in ('PENDING_USER_ACCEPTANCE', 'ACCEPTED_FOR_DESIGN_ONLY'):
        raise ValueError('Assumption acceptance must be explicit')
    return True
