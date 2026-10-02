"""Create reproducible diagnostic figures and measured error summaries."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .preprocess import load_data

def run(root='.'):
    root=Path(root); out=root/'results'; figs=out/'figures';figs.mkdir(exist_ok=True)
    data=load_data(root/'data/battery_modeling_ready.parquet')
    perf=pd.read_csv(out/'model_performance.csv')
    cv=pd.read_csv(out/'cv_comparison.csv')
    pred=pd.read_csv(out/'predictions.csv')
    sel=pred.loc[pred.model.eq('Selected') & pred.has_label & pred.split.ne('train')]
    plt.style.use('seaborn-v0_8-whitegrid')
    fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
    x=cv.loc[cv.model.ne('SelectionProcedure')].sort_values('CV_MAPE_mean')
    ax.barh(x.model,x.CV_MAPE_mean,xerr=x.CV_MAPE_SD,color='#2563eb',alpha=.8,capsize=3)
    ax.set(xlabel='Nested grouped CV MAPE (%) +/- fold SD',title='Fixed S2 candidate comparison (development cells only)')
    fig.savefig(figs/'cv_comparison.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for ax,split,title in zip(axes,['valid','test1','test2'],['Batch 1 Hold-out','Batch 2','Batch 3']):
        z=sel.loc[sel.split.eq(split)]
        ax.scatter(z.actual,z.predicted,color='#2563eb')
        lim=[min(z.actual.min(),z.predicted.min())*.9,max(z.actual.max(),z.predicted.max())*1.1]
        ax.plot(lim,lim,'--',color='#64748b');ax.set(xlim=lim,ylim=lim,xlabel='Actual cycle life',ylabel='Predicted cycle life',title=f'{title} (n={len(z)})')
    fig.savefig(figs/'actual_vs_predicted.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for split,c in zip(['valid','test1','test2'],['#2563eb','#ea580c','#059669']):
        z=sel.loc[sel.split.eq(split)]
        axes[0].scatter(z.actual,z.predicted-z.actual,color=c,label=split,alpha=.8)
    axes[0].axhline(0,color='#64748b',ls='--');axes[0].legend();axes[0].set(xlabel='Actual life',ylabel='Prediction - actual (cycles)',title='Residual pattern')
    axes[1].boxplot([sel.loc[sel.split.eq(s),'APE'] for s in ['valid','test1','test2']],tick_labels=['Valid','Batch 2','Batch 3'])
    axes[1].set(ylabel='Absolute percentage error (%)',title='Error distribution')
    fig.savefig(figs/'error_diagnostics.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,5),layout='constrained')
    table=perf.loc[perf.split.isin(['valid','test1','test2'])].pivot(index='model',columns='split',values='MAPE')
    table.plot.bar(ax=ax,color=['#ea580c','#059669','#2563eb']);ax.axhline(9.1,color='#dc2626',ls='--',label='Assignment reference 9.1%')
    ax.set(ylabel='MAPE (%)',title='Descriptive hold-out/test comparison - NOT a model selection criterion');ax.tick_params(axis='x',rotation=35);ax.legend()
    fig.savefig(figs/'external_comparison.png',dpi=180);plt.close(fig)
    errors=pd.read_csv(out/'error_analysis.csv')
    dev=data.loc[data.split.eq('train')]
    summary=[]
    for split,z in sel.groupby('split'):
        summary.append({'split':split,'n':len(z),'overpredicted':int((z.predicted>z.actual).sum()),
                        'median_residual':float((z.predicted-z.actual).median()),
                        'MAPE_short_lt500':float(z.loc[z.actual.lt(500),'APE'].mean()) if z.actual.lt(500).any() else None,
                        'MAPE_nonshort':float(z.loc[z.actual.ge(500),'APE'].mean()),
                        'short_lt500_n':int(z.actual.lt(500).sum())})
    pd.DataFrame(summary).to_csv(out/'error_summary.csv',index=False)
    ranges=[]
    for split in ['valid','test1','test2']:
        z=data.loc[data.split.eq(split)]
        for col in data.attrs['feature_sets']['S3']:
            lo,hi=dev[col].min(),dev[col].max()
            ranges.append({'split':split,'feature':col,'train_min':lo,'train_max':hi,'missing':int(z[col].isna().sum()),
                           'outside_train_range':int((z[col].lt(lo)|z[col].gt(hi)).sum()),'cells':len(z)})
    pd.DataFrame(ranges).to_csv(out/'feature_range_audit.csv',index=False)
    # Descriptive EDA with all batches, explicitly not used to retune the final model.
    eda=[]
    for batch,z in data.groupby('batch'):
        labeled=z.loc[z.has_label]
        corr=labeled[['DQ_logvar','Charge_time_median_early','C1','cycle_life']].corr(method='spearman')
        eda.append({'batch':batch,'n':len(z),'n_label':len(labeled),'life_min':labeled.cycle_life.min(),
                    'life_median':labeled.cycle_life.median(),'life_max':labeled.cycle_life.max(),
                    'short_lt500':int(labeled.cycle_life.lt(500).sum()),'long_gt1000':int(labeled.cycle_life.gt(1000).sum()),
                    'DQ_logvar_life_rho':corr.loc['DQ_logvar','cycle_life'],
                    'charge_time_life_rho':corr.loc['Charge_time_median_early','cycle_life'],
                    'C1_life_rho':corr.loc['C1','cycle_life']})
    pd.DataFrame(eda).to_csv(out/'eda_summary.csv',index=False)
    print('Figures and diagnostics saved:',figs)

if __name__=='__main__':run()
