import unittest
from pathlib import Path
import numpy as np
from src.preprocess import load_data
from src.features import derive_early_features, FEATURE_SETS
from src.train import build, metrics

class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df=load_data(Path(__file__).resolve().parents[1]/'data/battery_modeling_ready.parquet')

    def test_partition(self):
        self.assertEqual(self.df.groupby('split').size().to_dict(), {'test1':47,'test2':46,'train':37,'valid':9})
        self.assertEqual(int(self.df.has_label.sum()),129)

    def test_target_excluded(self):
        for fs in FEATURE_SETS.values():self.assertNotIn('cycle_life',fs)
        model,_=build('Ridge','S2')
        d=self.df.loc[self.df.split.eq('train')]
        model.fit(d,d.cycle_life)
        p=model.predict(d)
        changed=d.copy();changed['cycle_life']=1
        np.testing.assert_allclose(p,model.predict(changed))

    def test_derivation(self):
        d=derive_early_features(1,0.9,np.array([-0.1,-0.2]),np.array([3.,2.]),4.,2.)
        self.assertAlmostEqual(d['capacity_retention_100_pct'],90)
        self.assertAlmostEqual(d['C1_to_C2_ratio'],2)
        self.assertAlmostEqual(d['DQ_relative_loss_area'],0.15)

    def test_mape_units(self):
        self.assertAlmostEqual(metrics([100,200],[110,180])['MAPE'],10)

if __name__=='__main__':unittest.main()
