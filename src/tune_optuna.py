"""Fixed-budget Optuna TPE search with grouped nested CV and frozen external evaluation."""
import argparse
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import optuna
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import GroupKFold
from .features import FEATURE_SETS
from .preprocess import load_data, preprocessing
from .train import metrics, SEED

MODELS = ['SVR', 'CatBoost', 'RandomForest', 'ExtraTrees']
FS = dict(FEATURE_SETS)
FS['S4_early'] = FEATURE_SETS['S3'] + ['IR_change_10_100', 'Qd_slope_early',
                                     'Charge_time_median_early', 'DQ_relative_loss_area']


def propose(trial, name):
    trial.suggest_categorical('feature_set', list(FS))
    target = trial.suggest_categorical('target_transform', ['log', 'original'])
    if name == 'SVR':
        trial.suggest_categorical('kernel', ['rbf', 'linear'])
        trial.suggest_float('C', .01 if target == 'log' else 1, 100 if target == 'log' else 10000, log=True)
        trial.suggest_float('gamma', .001, 10, log=True)
        trial.suggest_float('epsilon_log' if target == 'log' else 'epsilon_original',
                            .001 if target == 'log' else 1, .2 if target == 'log' else 100, log=True)
    elif name == 'CatBoost':
        trial.suggest_int('iterations', 100, 500, step=100)
        trial.suggest_int('depth', 1, 5)
        trial.suggest_float('learning_rate', .005, .2, log=True)
        trial.suggest_float('l2_leaf_reg', .1, 50, log=True)
        trial.suggest_float('random_strength', 0, 3)
        trial.suggest_categorical('loss_function', ['RMSE', 'MAE'])
    else:
        trial.suggest_categorical('n_estimators', [100, 200])
        trial.suggest_int('max_depth', 1, 10)
        trial.suggest_int('min_samples_leaf', 1, 8)
        trial.suggest_float('max_features', .4, 1.)
        trial.suggest_categorical('criterion', ['squared_error', 'absolute_error'])
        trial.suggest_categorical('bootstrap', [False, True])
    return trial.params


def build(name, params):
    p = dict(params)
    fs = p.pop('feature_set')
    target = p.pop('target_transform')
    if name == 'SVR':
        p['epsilon'] = p.pop('epsilon_log' if target == 'log' else 'epsilon_original')
        p.pop('epsilon_original' if target == 'log' else 'epsilon_log', None)
        reg = SVR(**p)
    elif name == 'CatBoost':
        reg = CatBoostRegressor(**p, random_seed=SEED, thread_count=2, verbose=False,
                                allow_writing_files=False)
    else:
        cls = RandomForestRegressor if name == 'RandomForest' else ExtraTreesRegressor
        reg = cls(**p, random_state=SEED, n_jobs=2)
    if target == 'log':
        reg = TransformedTargetRegressor(regressor=reg, func=np.log, inverse_func=np.exp)
    selector = ColumnTransformer([('early', preprocessing(name == 'SVR'), FS[fs])], remainder='drop')
    return Pipeline([('features', selector), ('regressor', reg)])


def search(frame, name, stage, trials, out, seed):
    splits = list(GroupKFold(4).split(frame, groups=frame.group_id))
    for tr, va in splits:
        assert set(frame.iloc[tr].group_id).isdisjoint(frame.iloc[va].group_id)
    study = optuna.create_study(direction='minimize', study_name=f'{stage}_{name}',
                               storage=f'sqlite:///{(out / "studies.sqlite3").resolve()}',
                               sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=12,
                                                                  multivariate=False),
                               pruner=optuna.pruners.NopPruner())
    # Predefined simple starts; no outer/test labels or full-development winners injected.
    for fs in ['S1', 'S2', 'S3']:
        base = {'feature_set': fs, 'target_transform': 'log'}
        if name == 'SVR':
            base.update(kernel='rbf', C=1., gamma=.1, epsilon_log=.05)
        study.enqueue_trial(base)

    def objective(trial):
        params = propose(trial, name)
        scores = []
        for tr, va in splits:
            model = build(name, params)
            model.fit(frame.iloc[tr], frame.iloc[tr].cycle_life)
            scores.append(metrics(frame.iloc[va].cycle_life, model.predict(frame.iloc[va]))['MAPE'])
        trial.set_user_attr('fold_MAPE', scores)
        trial.set_user_attr('fold_SD', float(np.std(scores)))
        return float(np.mean(scores))

    study.optimize(objective, n_trials=trials, n_jobs=1)
    study.trials_dataframe().to_csv(out / f'{stage}_{name}_trials.csv', index=False)
    best = {'stage': stage, 'model': name, 'internal_CV_MAPE': study.best_value,
            'params': study.best_params, 'n_trials': len(study.trials)}
    model = build(name, study.best_params)
    model.fit(frame, frame.cycle_life)
    print(f'{stage}/{name}: {study.best_value:.3f}% / {study.best_params["feature_set"]} / '
          f'{study.best_params["target_transform"]}', flush=True)
    return model, best


def run(data, output, outer_trials=35, final_trials=80):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'studies.sqlite3').exists():
        raise FileExistsError('Choose a new --out directory to preserve existing studies.')
    start = time.time()
    df = load_data(data)
    dev = df.loc[df.split.eq('train') & df.has_label].reset_index(drop=True)
    assert len(dev) == 37
    rows, folds, oof = [], [], []
    for fold, (tr, va) in enumerate(GroupKFold(5).split(dev, groups=dev.group_id)):
        a, b = dev.iloc[tr], dev.iloc[va]
        assert set(a.group_id).isdisjoint(b.group_id)
        candidates = []
        for index, name in enumerate(MODELS):
            model, row = search(a, name, f'outer{fold}', outer_trials, out, SEED + 100*fold + index)
            candidates.append((model, row))
        selected = min(candidates, key=lambda item: item[1]['internal_CV_MAPE'])
        for label, (model, row) in list(zip(MODELS, candidates)) + [('SelectionProcedure', selected)]:
            pred = model.predict(b)
            folds.append({'fold': fold, 'model': label, 'selected_family': row['model'],
                          'feature_set': row['params']['feature_set'], **metrics(b.cycle_life, pred)})
            oof.extend({'fold': fold, 'model': label, 'battery_id': cell, 'actual': y, 'predicted': p}
                       for cell, y, p in zip(b.battery_id, b.cycle_life, pred))
        rows.extend(row for _, row in candidates)
        pd.DataFrame(folds).to_csv(out / 'nested_cv_folds.csv', index=False)
    final = [search(dev, name, 'final', final_trials, out, SEED + index)
             for index, name in enumerate(MODELS)]
    model, selected = min(final, key=lambda item: item[1]['internal_CV_MAPE'])
    selection = dict(selected, features=FS[selected['params']['feature_set']], seed=SEED,
                     outer_trials=outer_trials, final_trials=final_trials,
                     train_cells=37, train_policies=int(dev.group_id.nunique()),
                     holdout_used_for_selection=False, test_used_for_selection=False,
                     caveat='Exploration and previous model experiments already inspected these batches. '
                            'Nested CV within this new search does not eliminate prior analyst selection bias.')
    (out / 'selection.json').write_text(json.dumps(selection, indent=2))
    joblib.dump(model, out / 'final_model.joblib')
    candidate_dir = out / 'candidate_models'
    candidate_dir.mkdir(exist_ok=True)
    evaluation, predictions = [], []
    # All final candidates are frozen before external labels are evaluated.
    for label, (estimator, row) in list(zip(MODELS, final)) + [('Selected', (model, selected))]:
        joblib.dump(estimator, candidate_dir / f'{label}.joblib')
        for split in ['train', 'valid', 'test1', 'test2']:
            part = df.loc[df.split.eq(split)]
            p = estimator.predict(part)
            ok = part.has_label.to_numpy()
            evaluation.append({'model': label, 'feature_set': row['params']['feature_set'],
                               'target_transform': row['params']['target_transform'], 'split': split,
                               'n': int(ok.sum()), **metrics(part.loc[ok, 'cycle_life'], p[ok])})
            predictions.extend({'model': label, 'split': split, 'battery_id': cell,
                                'actual': y, 'predicted': value}
                               for cell, y, value in zip(part.battery_id, part.cycle_life, p))
    rows.extend(row for _, row in final)
    for name, records in [('best_trials', rows), ('oof_predictions', oof),
                          ('model_performance', evaluation), ('predictions', predictions)]:
        pd.DataFrame(records).to_csv(out / f'{name}.csv', index=False)
    oof_df = pd.DataFrame(oof)
    ranking = []
    for name, group in pd.DataFrame(folds).groupby('model'):
        vals = oof_df.loc[oof_df.model.eq(name)]
        ranking.append({'model': name, 'outer_CV_MAPE': group.MAPE.mean(),
                        'outer_CV_SD': group.MAPE.std(ddof=0),
                        'pooled_MAPE': metrics(vals.actual, vals.predicted)['MAPE'], 'n': len(vals)})
    rank = pd.DataFrame(ranking).sort_values('outer_CV_MAPE')
    rank.to_csv(out / 'cv_comparison.csv', index=False)
    summary = {'seconds': time.time()-start, 'optuna_version': optuna.__version__,
               'n_trials': 4*(5*outer_trials+final_trials), 'selection': selection,
               'cv': ranking, 'external': evaluation, 'feature_sets': FS,
               'objective': 'Mean original-space MAPE across four policy-disjoint folds',
               'pruning': False, 'study_parallelism': 1}
    (out / 'run_summary.json').write_text(json.dumps(summary, indent=2))
    print(rank.to_string(index=False), flush=True)
    print(pd.DataFrame(evaluation).loc[lambda d:d.model.eq('Selected')].to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/battery_modeling_ready.parquet')
    parser.add_argument('--out', default='results/optuna')
    parser.add_argument('--outer-trials', type=int, default=35)
    parser.add_argument('--final-trials', type=int, default=80)
    args = parser.parse_args()
    run(args.data, args.out, args.outer_trials, args.final_trials)
