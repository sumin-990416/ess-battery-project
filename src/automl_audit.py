"""Refit the already frozen inner-CV configurations for family-wise outer evaluation."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from flaml.automl.model import (LGBMEstimator, XGBoostEstimator, CatBoostEstimator,
                               RandomForestEstimator, ExtraTreesEstimator)
from .automl import ScaledLearner, ScaledElastic, ScaledSVR, LifePredictor
from .features import FEATURE_SETS
from .preprocess import load_data
from .train import metrics

CLASSES = {'scaled_ridge': ScaledLearner, 'scaled_elastic': ScaledElastic,
           'scaled_svr': ScaledSVR, 'lgbm': LGBMEstimator, 'xgboost': XGBoostEstimator,
           'catboost': CatBoostEstimator, 'rf': RandomForestEstimator,
           'extra_tree': ExtraTreesEstimator}


def audit(data='data/battery_modeling_ready.parquet', output='results/automl'):
    out = Path(output)
    df = load_data(data)
    dev = df.loc[df.split.eq('train') & df.has_label].reset_index(drop=True)
    table = pd.read_csv(out / 'leaderboard.csv')
    scores, predictions = [], []
    for fold, (tr, va) in enumerate(GroupKFold(5).split(dev, groups=dev.group_id)):
        a, b = dev.iloc[tr], dev.iloc[va]
        rows = table.loc[table.stage.eq(f'outer{fold}')].sort_values('internal_CV_MAPE')
        # For each family, pick features/config using ONLY this outer train's inner scores.
        for name, row in rows.groupby('model', sort=False).first().iterrows():
            params = json.loads(row.params)
            params.pop('FLAML_sample_size', None)
            estimator = CLASSES[name](task='regression', n_jobs=2, **params)
            estimator.fit(a[FEATURE_SETS[row.feature_set]], np.log(a.cycle_life.to_numpy()))
            model = LifePredictor(estimator, FEATURE_SETS[row.feature_set])
            p = model.predict(b)
            scores.append({'fold': fold, 'model': name, 'feature_set': row.feature_set,
                           'n': len(b), **metrics(b.cycle_life, p)})
            predictions.extend({'fold': fold, 'model': name, 'battery_id': cell,
                                'actual': y, 'predicted': v}
                               for cell, y, v in zip(b.battery_id, b.cycle_life, p))
        print(f'FAMILY AUDIT {fold+1}/5 completed', flush=True)
    scores = pd.DataFrame(scores)
    predictions = pd.DataFrame(predictions)
    scores.to_csv(out / 'family_nested_cv_folds.csv', index=False)
    predictions.to_csv(out / 'family_oof_predictions.csv', index=False)
    ranking = []
    for name, rows in scores.groupby('model'):
        pred = predictions.loc[predictions.model.eq(name)]
        ranking.append({'model': name, 'outer_CV_MAPE': rows.MAPE.mean(),
                        'outer_CV_SD': rows.MAPE.std(ddof=0), 'n': len(pred),
                        'pooled_MAPE': metrics(pred.actual, pred.predicted)['MAPE']})
    rank = pd.DataFrame(ranking).sort_values('outer_CV_MAPE')
    rank.to_csv(out / 'family_nested_cv_ranking.csv', index=False)
    external = []
    candidates_dir = out / 'candidate_models'
    candidates_dir.mkdir(exist_ok=True)
    final = table.loc[table.stage.eq('final')].sort_values('internal_CV_MAPE')
    for name, row in final.groupby('model', sort=False).first().iterrows():
        params = json.loads(row.params)
        params.pop('FLAML_sample_size', None)
        estimator = CLASSES[name](task='regression', n_jobs=2, **params)
        estimator.fit(dev[FEATURE_SETS[row.feature_set]], np.log(dev.cycle_life.to_numpy()))
        model = LifePredictor(estimator, FEATURE_SETS[row.feature_set])
        joblib.dump(model, candidates_dir / f'{name}.joblib')
        for split in ['valid', 'test1', 'test2']:
            part = df.loc[df.split.eq(split) & df.has_label]
            external.append({'model': name, 'feature_set': row.feature_set, 'split': split,
                             'n': len(part), **metrics(part.cycle_life, model.predict(part))})
    pd.DataFrame(external).to_csv(out / 'family_external_comparison.csv', index=False)
    print(rank.to_string(index=False), flush=True)
    return rank


if __name__ == '__main__':
    audit()
