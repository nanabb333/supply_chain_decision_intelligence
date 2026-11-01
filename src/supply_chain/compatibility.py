"""Organizer archive compatibility; source guide page 5 defines d_1 = 2011-01-29."""
import pandas as pd
ANCHOR = pd.Timestamp('2011-01-29')

def calendar_days(frame):
    result = frame.copy()
    dates = pd.to_datetime(result['date'], errors='raise')
    if dates.isna().any() or dates.duplicated().any():
        raise ValueError('Null or duplicate calendar dates')
    if 'd' not in result:
        if dates.min() != ANCHOR:
            raise ValueError('Organizer calendar must include authoritative d_1 anchor 2011-01-29')
        result['d'] = 'd_' + ((dates - ANCHOR).dt.days + 1).astype(str)
    if 'weekday' in result and not result.weekday.eq(dates.dt.day_name()).all():
        raise ValueError('Calendar weekday/date mismatch')
    for name, expected in [('year',dates.dt.year),('month',dates.dt.month),('wday',(dates.dt.dayofweek+2)%7+1)]:
        if name in result and not pd.to_numeric(result[name], errors='raise').eq(expected).all():
            raise ValueError(f'Calendar {name}/date mismatch')
    return result
