"""Grouped, nested FLAML search. Never select using hold-out/test targets."""
import argparse
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from flaml import AutoML, tune
from flaml.automl.model import SKLearnEstimator
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.model_selection import GroupKFold
from .features import FEATURE_SETS
from .preprocess import load_data
from .train import metrics, SEED


class ScaledLearner(SKLearnEstimator):
    kind = 'ridge'

    @classmethod
    def search_space(cls, data_size, **kwargs):
        if cls.kind == 'ridge':
            return {'alpha': {'domain': tune.loguniform(1e-4, 1e3), 'init_value': 1.0}}
        if cls.kind == 'elastic':
            return {'alpha': {'domain': tune.loguniform(1e-5, 1), 'init_value': .001},
                    'l1_ratio': {'domain': tune.uniform(.01, .99), 'init_value': .5}}
        return {'C': {'domain': tune.loguniform(.01, 100), 'init_value': 1.0},
                'gamma': {'domain': tune.loguniform(.001, 10), 'init_value': .1},
                'epsilon': {'domain': tune.loguniform(.001, .2), 'init_value': .05}}

    def fit(self, X_train, y_train, budget=None, **kwargs):
        start = time.time()
        params = {k: v for k, v in self.params.items() if k not in ['n_jobs', 'random_state']}
        reg = (Ridge(**params) if self.kind == 'ridge' else
               ElasticNet(**params, max_iter=100000, tol=1e-5) if self.kind == 'elastic' else
               SVR(**params))
        self._model = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True),
                                    StandardScaler(), reg)
        self._model.fit(X_train, y_train)
        return time.time() - start


class ScaledElastic(ScaledLearner):
    kind = 'elastic'


class ScaledSVR(ScaledLearner):
    kind = 'svr'


def original_mape(X_val, y_val, estimator, labels, X_train, y_train,
                  weight_val=None, weight_train=None, config=None,
                  groups_val=None, groups_train=None):
    score = metrics(np.exp(y_val), np.exp(estimator.predict(X_val)))['MAPE'] / 100
    return score, {'MAPE_percent': score * 100}


class LifePredictor:
    """Persisted model exposes predictions in cycles, not log(cycles)."""
    def __init__(self, estimator, features):
        self.estimator = estimator
        self.features = list(features)

    def predict(self, frame):
        return np.exp(self.estimator.predict(frame[self.features]))


LEARNERS = ['scaled_ridge', 'scaled_elastic', 'scaled_svr',
            'lgbm', 'xgboost', 'catboost', 'rf', 'extra_tree']


def search(frame, fs, seconds, out, stage):
    X = frame[FEATURE_SETS[fs]].copy()
    assert np.isfinite(X.to_numpy()).all(), 'Missing/nonfinite values require fold-local handling.'
    cv = GroupKFold(4)
    for tr, va in cv.split(X, groups=frame.group_id):
        assert set(frame.iloc[tr].group_id).isdisjoint(frame.iloc[va].group_id)
    automl = AutoML()
    automl.add_learner('scaled_ridge', ScaledLearner)
    automl.add_learner('scaled_elastic', ScaledElastic)
    automl.add_learner('scaled_svr', ScaledSVR)
    started = time.time()
    automl.fit(X_train=X, y_train=np.log(frame.cycle_life.to_numpy()),
               task='regression', metric=original_mape, eval_method='cv',
               split_type=cv, groups=frame.group_id.to_numpy(), n_splits=4,
               estimator_list=LEARNERS, learner_selector='roundrobin',
               time_budget=seconds, max_iter=160, sample=False, skip_transform=True,
               n_jobs=2, seed=SEED, verbose=0, model_history=True,
               log_file_name=str(out / f'{stage}_{fs}.log'), log_type='all')
    rows = []
    for name, loss in automl.best_loss_per_estimator.items():
        if np.isfinite(loss):
            rows.append({'stage': stage, 'feature_set': fs, 'model': name,
                         'internal_CV_MAPE': 100 * loss,
                         'params': json.dumps(automl.best_config_per_estimator[name], default=str),
                         'seconds': time.time() - started})
    best = {'model': automl.best_estimator, 'feature_set': fs,
            'internal_CV_MAPE': 100 * automl.best_loss,
            'params': automl.best_config, 'seconds': time.time() - started}
    print(f'{stage}/{fs}: {best["model"]}, inner MAPE={best["internal_CV_MAPE"]:.3f}%, '
          f'{len(rows)}/{len(LEARNERS)} families searched', flush=True)
    return LifePredictor(automl.model, FEATURE_SETS[fs]), best, rows


def select(frame, seconds, out, stage):
    candidates = [search(frame, fs, seconds, out, stage) for fs in ['S1', 'S2', 'S3']]
    winner = min(candidates, key=lambda x: x[1]['internal_CV_MAPE'])
    return winner[0], winner[1], [row for _, _, rows in candidates for row in rows]


def run(data_path, output, seconds=20, smoke=False):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    df = load_data(data_path)
    dev = df.loc[df.split.eq('train') & df.has_label].reset_index(drop=True)
    assert len(dev) == 37 and dev.group_id.nunique() == 18
    if smoke:
        model, best, rows = search(dev, 'S2', seconds, out, 'smoke')
        print(best, model.predict(dev)[:3], flush=True)
        return
    start = time.time()
    folds, oof, leaderboard = [], [], []
    for fold, (tr, va) in enumerate(GroupKFold(5).split(dev, groups=dev.group_id)):
        a, b = dev.iloc[tr], dev.iloc[va]
        assert set(a.group_id).isdisjoint(b.group_id)
        model, best, rows = select(a, seconds, out, f'outer{fold}')
        leaderboard.extend(rows)
        pred = model.predict(b)
        folds.append({'fold': fold, 'model': best['model'], 'feature_set': best['feature_set'],
                      'n': len(b), **metrics(b.cycle_life, pred)})
        for cell, y, p in zip(b.battery_id, b.cycle_life, pred):
            oof.append({'fold': fold, 'battery_id': cell, 'actual': y, 'predicted': p})
        pd.DataFrame(folds).to_csv(out / 'nested_cv_folds.csv', index=False)
    model, best, rows = select(dev, seconds, out, 'final')
    leaderboard.extend(rows)
    # Freeze winner before examining any external target.
    best.update(features=model.features, seed=SEED, train_cells=len(dev),
                train_policies=dev.group_id.nunique(), budget_per_feature_seconds=seconds,
                holdout_used_for_selection=False, test_used_for_selection=False,
                caveat='Previous EDA and prior experiments already used these batches; '
                       'they are not pristine untouched tests. Time-budgeted search can vary across runs.')
    (out / 'selection.json').write_text(json.dumps(best, indent=2, default=str))
    joblib.dump(model, out / 'final_model.joblib')
    assert np.allclose(joblib.load(out / 'final_model.joblib').predict(dev), model.predict(dev))
    scores, predictions = [], []
    for split in ['train', 'valid', 'test1', 'test2']:
        part = df.loc[df.split.eq(split)]
        p = model.predict(part)
        ok = part.has_label.to_numpy()
        scores.append({'split': split, 'n': int(ok.sum()), 'model': best['model'],
                       'feature_set': best['feature_set'], **metrics(part.loc[ok, 'cycle_life'], p[ok])})
        for cell, y, value in zip(part.battery_id, part.cycle_life, p):
            predictions.append({'split': split, 'battery_id': cell, 'actual': y, 'predicted': value})
    fold_df = pd.DataFrame(folds)
    summary = {'nested_CV_MAPE_mean': fold_df.MAPE.mean(),
               'nested_CV_MAPE_SD': fold_df.MAPE.std(ddof=0),
               'pooled_OOF_MAPE': metrics(pd.DataFrame(oof).actual, pd.DataFrame(oof).predicted)['MAPE'],
               'seconds': time.time() - start, 'learner_list': LEARNERS,
               'feature_sets': ['S1', 'S2', 'S3'], 'n_jobs': 2,
               'selection': best, 'external': scores}
    for name, rows in [('leaderboard', leaderboard), ('oof_predictions', oof),
                       ('model_performance', scores), ('predictions', predictions)]:
        pd.DataFrame(rows).to_csv(out / f'{name}.csv', index=False)
    (out / 'run_summary.json').write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str), flush=True)
    from .automl_audit import audit
    audit(data_path, out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/battery_modeling_ready.parquet')
    parser.add_argument('--out', default='results/automl')
    parser.add_argument('--seconds', type=float, default=20)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    # Import canonical module so joblib never stores __main__ class references.
    from src.automl import run as canonical_run
    canonical_run(args.data, args.out, args.seconds, args.smoke)
