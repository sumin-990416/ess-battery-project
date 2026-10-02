"""Retrain the frozen SVR without tuning; preserve the original partition."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.model_selection import GroupKFold
from .preprocess import load_data, preprocessing
from .train import metrics


def build_frozen(selection):
    selector = ColumnTransformer([('features', preprocessing(True), selection['features'])], remainder='drop')
    target = TransformedTargetRegressor(regressor=SVR(kernel='rbf', **selection['params']),
                                       func=np.log, inverse_func=np.exp)
    return Pipeline([('features', selector), ('model', target)])


def run(data='data/battery_modeling_ready.parquet', selection_path='results/automl/selection.json',
        output='results/final_svr'):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'run_summary.json').exists():
        raise FileExistsError('Use a new --out to retain earlier final runs.')
    selection = json.loads(Path(selection_path).read_text())
    df = load_data(data)
    dev = df.loc[df.split.eq('train') & df.has_label].reset_index(drop=True)
    assert len(dev) == 37 and dev.group_id.nunique() == 18
    estimator = build_frozen(selection)
    cv_rows, oof = [], []
    for n_splits in [4, 5]:
        for fold, (tr, va) in enumerate(GroupKFold(n_splits).split(dev, groups=dev.group_id)):
            a, b = dev.iloc[tr], dev.iloc[va]
            assert set(a.group_id).isdisjoint(b.group_id)
            fitted = clone(estimator).fit(a, a.cycle_life)
            p = fitted.predict(b)
            cv_rows.append({'n_splits': n_splits, 'fold': fold, 'n': len(b), **metrics(b.cycle_life, p)})
            oof.extend({'n_splits': n_splits, 'fold': fold, 'battery_id': cell, 'actual': y, 'predicted': value}
                       for cell, y, value in zip(b.battery_id, b.cycle_life, p))
    model = estimator.fit(dev, dev.cycle_life)
    refit = clone(estimator).fit(dev, dev.cycle_life)
    np.testing.assert_allclose(model.predict(df), refit.predict(df), atol=1e-10, rtol=1e-10)
    changed = df.copy()
    changed['cycle_life'] = 1
    np.testing.assert_allclose(model.predict(df), model.predict(changed))
    joblib.dump(model, out / 'final_model.joblib')
    np.testing.assert_allclose(model.predict(df), joblib.load(out / 'final_model.joblib').predict(df))
    scores, predictions = [], []
    for split in ['train', 'valid', 'test1', 'test2']:
        part = df.loc[df.split.eq(split)]
        pred = model.predict(part)
        labeled = part.has_label.to_numpy()
        scores.append({'split': split, 'n_labeled': int(labeled.sum()), 'n_predicted': len(part),
                       **metrics(part.loc[labeled, 'cycle_life'], pred[labeled])})
        for (_, cell), value in zip(part.iterrows(), pred):
            predictions.append({'split': split, 'battery_id': cell.battery_id, 'policy': cell.policy,
                                'actual': cell.cycle_life, 'predicted': value,
                                'residual': value-cell.cycle_life if cell.has_label else np.nan,
                                'APE': 100*abs(value-cell.cycle_life)/cell.cycle_life if cell.has_label else np.nan})
    scores = pd.DataFrame(scores)
    folds = pd.DataFrame(cv_rows)
    pred = pd.DataFrame(predictions)
    previous_path = Path(selection_path).parent / 'final_model.joblib'
    previous_pred = joblib.load(previous_path).predict(df)
    cv = [{'n_splits': n, 'mean_MAPE': g.MAPE.mean(), 'fold_SD': g.MAPE.std(ddof=0),
           'caveat': 'Fixed-parameter CV after selection on these development cells; not an independent nested estimate.'}
          for n, g in folds.groupby('n_splits')]
    previous_run = json.loads((Path(selection_path).parent / 'run_summary.json').read_text())
    summary = {'frozen_parameters': dict(kernel='rbf', **selection['params']),
               'features': selection['features'], 'target_transform': 'log/exp',
               'train_cells': len(dev), 'train_policies': dev.group_id.nunique(),
               'no_retuning': True, 'holdout_used_for_training': False,
               'test_used_for_training_or_selection': False, 'fixed_CV': cv,
               'previous_selection_procedure_nested_MAPE': previous_run['nested_CV_MAPE_mean'],
               'previous_selection_procedure_nested_SD': previous_run['nested_CV_MAPE_SD'],
               'max_abs_prediction_difference_vs_previous': float(np.max(abs(model.predict(df)-previous_pred))),
               'checks': ['two independent fits agree', 'joblib reload agrees',
                          'target column excluded', 'all grouped folds policy-disjoint'],
               'metrics': scores.to_dict('records'),
               'caveat': 'Same previously inspected datasets; rerunning does not constitute new external validation. '
                         'Provided cycle_life/EOL and original-paper data mapping remain unverified.'}
    for name, frame in [('model_performance', scores), ('cv_folds', folds),
                        ('oof_predictions', pd.DataFrame(oof)), ('predictions', pred),
                        ('error_analysis', pred.loc[pred.split.ne('train') & pred.actual.notna()].sort_values('APE', ascending=False))]:
        frame.to_csv(out / f'{name}.csv', index=False)
    pd.DataFrame(cv).to_csv(out / 'fixed_cv_summary.csv', index=False)
    s = scores.set_index('split')
    nested = summary['previous_selection_procedure_nested_MAPE']
    report = [('Train (Batch 1 CV, prior nested selection)', nested, '%', 37),
              ('Valid (Batch 1 Hold-out)', s.loc['valid', 'MAPE'], '%', 9),
              ('Test (Batch 2)', s.loc['test1', 'MAPE'], '%', 39),
              ('Gap (Train-Valid)', s.loc['valid','MAPE']-nested, 'pp', None),
              ('Gap (Valid-Test)', s.loc['test1','MAPE']-s.loc['valid','MAPE'], 'pp', None),
              ('Gap (Target-Test): Batch 2', s.loc['test1','MAPE']-9.1, 'pp', None),
              ('Test (Batch 3)', s.loc['test2','MAPE'], '%', 44),
              ('Gap (Batch2-Batch3)', s.loc['test2','MAPE']-s.loc['test1','MAPE'], 'pp', None)]
    pd.DataFrame(report, columns=['Index','value','unit','n']).to_csv(out / 'performance_reporting.csv', index=False)
    (out / 'run_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/battery_modeling_ready.parquet')
    parser.add_argument('--selection', default='results/automl/selection.json')
    parser.add_argument('--out', default='results/final_svr')
    args = parser.parse_args()
    run(args.data, args.selection, args.out)
