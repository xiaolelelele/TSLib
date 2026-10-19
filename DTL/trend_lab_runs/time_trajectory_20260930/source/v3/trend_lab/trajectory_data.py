"""Explicit chronological TIME-export pilot, preserving exact float64 labels."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from .data import Split


def load_trajectory_data(spec):
    path = Path(spec['csv'])
    frame = pd.read_csv(path,float_precision='round_trip')
    times = pd.to_datetime(frame.pop('date'))
    if times.duplicated().any() or not times.is_monotonic_increasing:
        raise ValueError('timestamps must be unique and increasing')
    step = pd.Timedelta(spec['frequency'])
    if not (times.diff().iloc[1:] == step).all():
        raise ValueError('keep every raw time slot; do not compress gaps')
    raw = frame.to_numpy(dtype=np.float64)
    if np.isinf(raw).any():
        raise ValueError('infinite observations')
    tz = times.dt.tz
    boundaries = [pd.Timestamp(v, tz=tz) for v in spec['boundaries']]
    cuts = [int(times.searchsorted(v)) for v in boundaries]
    if len(cuts) != 6 or cuts[0] != 0 or cuts[-1] != len(raw) or any(a >= b for a,b in zip(cuts[:-1],cuts[1:])):
        raise ValueError('need five nonempty explicit target intervals covering exported data')
    train = raw[:cuts[1]]
    mean, std = np.nanmean(train,0), np.nanstd(train,0)
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std < 1e-8):
        raise ValueError('training statistics invalid')
    values = (raw - mean) / std
    valid = np.isfinite(values).all(1)
    bad = np.r_[0,np.cumsum(~valid)]
    history_policy=spec.get('history_policy','complete')
    if history_policy not in ('complete','causal_ffill'):
        raise ValueError('unknown history policy')
    # These filled values are for historical covariates ONLY. True target
    # values, training statistics, and real origin anchors remain untouched.
    historical=(pd.DataFrame(values).ffill().to_numpy() if history_policy=='causal_ffill' else values)
    history_bad=np.r_[0,np.cumsum(~np.isfinite(historical).all(1))]
    lookback,horizon = spec['lookback'],spec['horizon']
    splits, rows = {},{}
    for name,start,end in zip(['train','earlystop','calibration','validation','test'],cuts[:-1],cuts[1:]):
        stride = spec['train_stride'] if name=='train' else spec['evaluation_stride']
        origins = np.arange(0,len(raw),stride)
        eligible = origins[(origins >= lookback-1)&(origins+1 >= start)&(origins+horizon+1 <= end)]
        keep = ((bad[eligible+horizon+1] == bad[eligible+1]) & valid[eligible] &
                (history_bad[eligible+1] == history_bad[eligible-lookback+1]))
        chosen = eligible[keep]
        if len(chosen) < (300 if name=='train' else 40):
            raise ValueError('%s insufficient complete windows: %d'%(name,len(chosen)))
        x=np.stack([historical[t-lookback+1:t+1] for t in chosen])
        y=np.stack([values[t+1:t+horizon+1] for t in chosen])
        splits[name]=Split(x,y,chosen)
        rows[name]={'bounds':[start,end],'declared_dates':[str(boundaries[list(splits).index(name)]),str(boundaries[list(splits).index(name)+1])],
                    'eligible':len(eligible),'kept':len(chosen),'discarded':int((~keep).sum()),
                    'first_origin':str(times.iloc[chosen[0]]),'last_origin':str(times.iloc[chosen[-1]]),
                    'origin_hours':sorted(times.iloc[chosen].dt.hour.unique().tolist()),
                    'origin_minutes':sorted(times.iloc[chosen].dt.minute.unique().tolist()),
                    'filled_history_fraction':float(np.mean(np.stack([~np.isfinite(values[t-lookback+1:t+1]) for t in chosen])))}
    finite_train=np.isfinite(train)
    native=bool(np.all(train[finite_train] == np.rint(train[finite_train])))
    if spec['grid_mode']=='native' and not native:
        raise ValueError('declared integer count grid disagrees with training values')
    minimum=np.nanmin(values[:cuts[1]],0);maximum=np.nanmax(values[:cuts[1]],0)
    metadata={'csv':str(path.resolve()),'csv_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
              'columns':frame.columns.tolist(),'mean':mean.tolist(),'std':std.tolist(),
              'train_min':minimum.tolist(),'train_max':maximum.tolist(),'native_integer_training':native,
              'cuts':cuts,'splits':rows,'lookback':lookback,'horizon':horizon,'spec':spec,
              'timestamp_axis':str(tz) if tz else 'source naive; timezone not established',
              'normalization':'float64 raw and train-only stats; torch inputs float32; evaluation labels float64',
              'processed_data_only':True,'upstream_cleaning_mask_available':False}
    metadata['history_policy']=history_policy
    return splits,metadata,times
