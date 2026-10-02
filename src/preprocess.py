from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def load_data(path):
    df = pd.read_parquet(Path(path))
    assert df.battery_id.is_unique
    assert set(df.split) == {'train', 'valid', 'test1', 'test2'}
    assert set(df.loc[df.split.eq('train'), 'group_id']).isdisjoint(
        df.loc[df.split.eq('valid'), 'group_id'])
    assert df.has_label.equals(df.cycle_life.notna())
    assert (df.loc[df.has_label, 'cycle_life'] > 0).all()
    assert not np.isinf(df[df.attrs['feature_columns']].to_numpy()).any()
    return df

def preprocessing(scale=True):
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median', keep_empty_features=True)),
        ('scaler', StandardScaler() if scale else 'passthrough'),
    ])
