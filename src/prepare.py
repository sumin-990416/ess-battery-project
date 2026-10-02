"""Optional full raw .mat -> early features -> fixed split regeneration."""
import argparse, os, runpy, contextlib, io
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from .features import derive_early_features, FEATURE_SETS

def run(raw_dir, output):
    os.environ['BATTERY_DATA_DIR']=str(Path(raw_dir).resolve())
    import IPython.display
    original_display,original_show=IPython.display.display,plt.show
    try:
        IPython.display.display=lambda *a,**k:None
        plt.show=lambda:plt.close('all')
        with contextlib.redirect_stdout(io.StringIO()):
            g=runpy.run_path(str(Path(__file__).with_name('raw_eda.py')))
    finally:
        IPython.display.display=original_display;plt.show=original_show
    rows=[]
    for batch,cells in g['batches'].items():
        for cell in cells:
            f={'batch':batch, 'battery':cell['battery'], 'policy':cell['policy'],
               'cycle_life':cell['life'], **cell['features']}
            f.update(derive_early_features(f['Qd_10'],f['Qd_100'],cell['delta'],
                                          cell['voltage'],f['C1'],f['C2']))
            rows.append(f)
    df=pd.DataFrame(rows)
    df['split']=df.batch.map({'Batch 1':'train','Batch 2':'test1','Batch 3':'test2'})
    df['source_split']=df['split']
    df['has_label']=df.cycle_life.notna()
    df['battery_id']=[f"B{b.rsplit(' ',1)[-1]}_{int(i):03d}" for b,i in zip(df.batch,df.battery)]
    df['group_id']=df.batch+'::'+df.policy
    idx=df.index[df.batch.eq('Batch 1')].to_numpy()
    tr,va=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=20261001).split(df.loc[idx],groups=df.loc[idx,'policy']))
    dev=idx[tr];df.loc[idx[va],'split']='valid'
    df['outer_cv_fold']=pd.Series(pd.NA,index=df.index,dtype='Int64')
    for fold,(_,v) in enumerate(GroupKFold(5).split(df.loc[dev],groups=df.loc[dev,'group_id'])):
        df.loc[dev[v],'outer_cv_fold']=fold
    df.attrs={'feature_columns':g['feature_cols']+['DQ_relative_logvar','DQ_relative_loss_area','capacity_retention_100_pct','C1_to_C2_ratio'],
              'feature_sets':FEATURE_SETS,'target_column':'cycle_life','group_column':'group_id','split_column':'split',
              'cutoff_cycle':100,'batch_mapping':g['BATCH_FILES'],
              'definitions':{'DQ_relative_logvar':'log10(var(delta/Q10))','DQ_relative_loss_area':'mean voltage integral max(-delta/Q10,0)',
                             'capacity_retention_100_pct':'100*Q100/Q10','C1_to_C2_ratio':'C1/C2'}}
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    df.to_parquet(output,index=False)
    print('Raw reconstruction saved:',output,df.shape)
    return df

if __name__=='__main__':
    load_dotenv(Path(__file__).resolve().parents[1]/'.env',override=False)
    p=argparse.ArgumentParser();p.add_argument('--raw',default=os.environ.get('BATTERY_DATA_DIR'));p.add_argument('--output',default='data/battery_modeling_ready.parquet')
    a=p.parse_args()
    if not a.raw:
        p.error('Set BATTERY_DATA_DIR in .env/environment or provide --raw.')
    run(a.raw,a.output)
