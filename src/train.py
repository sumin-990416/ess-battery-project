"""Run the fixed development-only selection procedure; evaluate hold-outs afterwards."""
import argparse, json, time, sys, platform
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import sklearn
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, r2_score
from .preprocess import load_data, preprocessing
from .features import FEATURE_SETS

SEED = 20261001
MODELS = ['Dummy', 'LinearRegression', 'Ridge', 'ElasticNet', 'SVR',
          'RandomForest', 'ExtraTrees', 'GradientBoosting']

def metrics(y, p):
    return {'MAPE': 100*mean_absolute_percentage_error(y, p),
            'MAE': mean_absolute_error(y, p),
            'RMSE': float(np.sqrt(mean_squared_error(y, p))), 'R2': r2_score(y, p)}

def build(name, feature_set):
    scale = name in ['LinearRegression', 'Ridge', 'ElasticNet', 'SVR']
    estimators = {
        'Dummy': DummyRegressor(strategy='median'),
        'LinearRegression': LinearRegression(), 'Ridge': Ridge(),
        'ElasticNet': ElasticNet(max_iter=100000, tol=1e-5),
        'SVR': SVR(epsilon=0.05),
        'RandomForest': RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=1),
        'ExtraTrees': ExtraTreesRegressor(n_estimators=300, random_state=SEED, n_jobs=1),
        'GradientBoosting': GradientBoostingRegressor(random_state=SEED, min_samples_leaf=5),
    }
    grids = {
        'Dummy': {}, 'LinearRegression': {},
        'Ridge': {'alpha': [0.01, 0.1, 1, 10, 100]},
        'ElasticNet': {'alpha': [0.001, 0.01, 0.1, 1], 'l1_ratio': [0.1, 0.5, 0.9]},
        'SVR': {'C': [0.1, 1, 10], 'gamma': ['scale', 0.1, 1]},
        'RandomForest': {'max_depth': [2, 3, None], 'min_samples_leaf': [3, 5], 'max_features': [0.7, 1.]},
        'ExtraTrees': {'max_depth': [2, 3, None], 'min_samples_leaf': [3, 5], 'max_features': [0.7, 1.]},
        'GradientBoosting': {'n_estimators': [50, 100], 'learning_rate': [0.03, 0.1], 'max_depth': [1, 2]},
    }
    reg = estimators[name]
    if name != 'Dummy':
        reg = TransformedTargetRegressor(regressor=reg, func=np.log, inverse_func=np.exp)
    selector = ColumnTransformer([('features', preprocessing(scale), FEATURE_SETS[feature_set])], remainder='drop')
    estimator = Pipeline([('features', selector), ('model', reg)])
    prefix = 'model__' if name == 'Dummy' else 'model__regressor__'
    return estimator, {prefix+k: v for k, v in grids[name].items()}

def fit_search(df, name, feature_set, n_jobs):
    estimator, grid = build(name, feature_set)
    cv = list(GroupKFold(4).split(df, groups=df.group_id))
    for tr, va in cv:
        assert set(df.iloc[tr].group_id).isdisjoint(df.iloc[va].group_id)
    search = GridSearchCV(estimator, grid, scoring='neg_mean_absolute_percentage_error',
                          cv=cv, n_jobs=n_jobs, error_score='raise', refit=True)
    search.fit(df, df.cycle_life)
    row = {'model': name, 'feature_set': feature_set,
           'internal_CV_MAPE': -100*search.best_score_,
           'internal_CV_SD': 100*search.cv_results_['std_test_score'][search.best_index_],
           'params': json.dumps(search.best_params_, ensure_ascii=False)}
    return search.best_estimator_, row

def select(df, n_jobs, label):
    rows, models = [], {}
    for name in MODELS:
        model, row = fit_search(df, name, 'S2', n_jobs)
        rows.append(row); models[(name, 'S2')] = model
        print(f'{label}: {name} S2 CV={row["internal_CV_MAPE"]:.3f}%', flush=True)
    top = sorted([r for r in rows if r['model'] != 'Dummy'], key=lambda r:r['internal_CV_MAPE'])[:3]
    for item in top:
        for fs in ['S1', 'S3', 'S2_plus_ratio', 'S2_original_DQ']:
            model, row = fit_search(df, item['model'], fs, n_jobs)
            rows.append(row); models[(item['model'], fs)] = model
    # Exact lowest development CV MAPE, predefined deterministic tie break by complexity.
    rank = {m:i for i,m in enumerate(MODELS)}
    best = min(rows, key=lambda r:(r['internal_CV_MAPE'], len(FEATURE_SETS[r['feature_set']]), rank[r['model']]))
    print(f'{label}: SELECTED {best["model"]}/{best["feature_set"]} CV={best["internal_CV_MAPE"]:.3f}%', flush=True)
    return models[(best['model'], best['feature_set'])], best, rows, models

def run(data_path, out, n_jobs=2):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    all_df = load_data(data_path)
    dev = all_df.loc[all_df.split.eq('train') & all_df.has_label].copy().reset_index(drop=True)
    assert len(dev)==37 and dev.group_id.nunique()==18
    start = time.time()
    outer = list(GroupKFold(5).split(dev, groups=dev.group_id))
    fold_rows, trial_rows, oof_rows = [], [], []
    for fold, (tr, va) in enumerate(outer):
        a, b = dev.iloc[tr].copy(), dev.iloc[va].copy()
        assert set(a.group_id).isdisjoint(b.group_id)
        assert set(b.outer_cv_fold.astype(int)) == {fold}
        model, best, trials, models = select(a, n_jobs, f'OUTER {fold+1}/5')
        trial_rows.extend(dict(r, stage='outer', fold=fold) for r in trials)
        pred = model.predict(b)
        fold_rows.append(dict(fold=fold, model=best['model'], feature_set=best['feature_set'],
                              n=len(b), **metrics(b.cycle_life, pred)))
        # All 8 fixed-S2 candidates receive nested tuning, independently of outer labels.
        for name in MODELS:
            p = models[(name, 'S2')].predict(b)
            for i, value in zip(b.index, p):
                oof_rows.append({'model': name, 'feature_set': 'S2', 'fold': fold,
                                 'battery_id': dev.loc[i,'battery_id'], 'actual': dev.loc[i,'cycle_life'], 'predicted': value})
        for i, value in zip(b.index, pred):
            oof_rows.append({'model': 'SelectionProcedure', 'feature_set': best['feature_set'], 'fold': fold,
                             'battery_id': dev.loc[i,'battery_id'], 'actual': dev.loc[i,'cycle_life'], 'predicted': value})
        pd.DataFrame(fold_rows).to_csv(out/'nested_cv_folds.csv', index=False)
    selected, best, trials, models = select(dev, n_jobs, 'FINAL DEVELOPMENT')
    trial_rows.extend(dict(r, stage='final', fold=-1) for r in trials)
    pd.DataFrame(trial_rows).to_csv(out/'tuning_trials.csv', index=False)
    oof = pd.DataFrame(oof_rows)
    oof.to_csv(out/'oof_predictions.csv', index=False)
    cv_rows = []
    for name, z in oof.groupby('model', sort=False):
        folds = [metrics(w.actual, w.predicted)['MAPE'] for _, w in z.groupby('fold')]
        cv_rows.append(dict(model=name, CV_MAPE_mean=np.mean(folds), CV_MAPE_SD=np.std(folds, ddof=0),
                            n=len(z), pooled_MAPE=metrics(z.actual,z.predicted)['MAPE']))
    cv = pd.DataFrame(cv_rows); cv.to_csv(out/'cv_comparison.csv',index=False)
    # Freeze selection before evaluating any valid/test target.
    selection = dict(best, features=FEATURE_SETS[best['feature_set']], train_cells=len(dev), train_policies=18,
                     seed=SEED, selection_rule='Lowest development-only inner grouped-CV MAPE; exact ties prefer fewer features and simpler model.',
                     holdout_used_for_selection=False, test_used_for_selection=False,
                     environment={'python':platform.python_version(),'sklearn':sklearn.__version__},
                     cv_caveat='Whole-dataset EDA preceded split design; nested CV cannot remove that prior exploratory selection bias.')
    (out/'selection.json').write_text(json.dumps(selection,ensure_ascii=False,indent=2))
    joblib.dump(selected, out/'final_model.joblib')
    evaluation_models = {(name,'S2'): models[(name,'S2')] for name in MODELS}
    evaluation_models[('Selected',best['feature_set'])] = selected
    eval_rows, prediction_rows = [], []
    for (name, fs), model in evaluation_models.items():
        for split in ['train','valid','test1','test2']:
            part = all_df.loc[all_df.split.eq(split)]
            pred = model.predict(part)
            labeled = part.has_label.to_numpy()
            eval_rows.append(dict(model=name, feature_set=fs, split=split, n=int(labeled.sum()),
                                  n_prediction=len(part), **metrics(part.loc[labeled,'cycle_life'], pred[labeled])))
            for (_, cell), value in zip(part.iterrows(), pred):
                prediction_rows.append(dict(model=name, feature_set=fs, split=split,
                     battery_id=cell.battery_id, batch=cell.batch, policy=cell.policy,
                     actual=cell.cycle_life, predicted=value, has_label=cell.has_label,
                     abs_error=abs(cell.cycle_life-value) if cell.has_label else np.nan,
                     APE=100*abs(cell.cycle_life-value)/cell.cycle_life if cell.has_label else np.nan))
    performance=pd.DataFrame(eval_rows)
    performance.to_csv(out/'model_performance.csv',index=False)
    predictions=pd.DataFrame(prediction_rows)
    predictions.to_csv(out/'predictions.csv',index=False)
    row=performance.loc[performance.model.eq('Selected')].set_index('split')
    train_cv=cv.loc[cv.model.eq('SelectionProcedure'),'CV_MAPE_mean'].iloc[0]
    reporting=[('Train (Batch 1 CV)',train_cv,'%',37),('Valid (Batch 1 Hold-out)',row.loc['valid','MAPE'],'%',9),
        ('Test (Batch 2)',row.loc['test1','MAPE'],'%',39),
        ('Gap (Train-Valid)',row.loc['valid','MAPE']-train_cv,'pp',None),
        ('Gap (Valid-Test)',row.loc['test1','MAPE']-row.loc['valid','MAPE'],'pp',None),
        ('Gap (Target-Test): Batch 2',row.loc['test1','MAPE']-9.1,'pp',None),
        ('Test (Batch 3)',row.loc['test2','MAPE'],'%',44),
        ('Gap (Batch2-Batch3)',row.loc['test2','MAPE']-row.loc['test1','MAPE'],'pp',None),
        ('Gap (Target-Test): Batch 3',row.loc['test2','MAPE']-9.1,'pp',None)]
    pd.DataFrame(reporting,columns=['Index','value','unit','n']).to_csv(out/'performance_reporting.csv',index=False)
    worst=predictions.loc[predictions.model.eq('Selected') & predictions.has_label & predictions.split.ne('train')]
    worst=worst.merge(all_df[['battery_id']+FEATURE_SETS['S3']],on='battery_id',validate='one_to_one')
    for col in FEATURE_SETS['S3']:
        lo,hi=dev[col].min(),dev[col].max()
        worst[col+'_outside_train_range']=worst[col].lt(lo)|worst[col].gt(hi)
    worst.sort_values('APE',ascending=False).to_csv(out/'error_analysis.csv',index=False)
    (out/'run_summary.json').write_text(json.dumps({'runtime_seconds':time.time()-start,'selected':selection,
                           'split_counts':all_df.groupby('split').size().to_dict()},ensure_ascii=False,indent=2))
    print('DONE',json.dumps(selection,ensure_ascii=False),flush=True)
    print(performance.loc[performance.model.eq('Selected')].to_string(index=False),flush=True)
    return selection

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data',default='data/battery_modeling_ready.parquet')
    p.add_argument('--output',default='results');p.add_argument('--jobs',type=int,default=2)
    a=p.parse_args();run(a.data,a.output,a.jobs)
