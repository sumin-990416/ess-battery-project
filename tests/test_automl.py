import unittest
from pathlib import Path
import tempfile
import joblib
import numpy as np
from sklearn.model_selection import GroupKFold
from src.automl import ScaledSVR, LifePredictor, original_mape
from src.preprocess import load_data
from src.features import FEATURE_SETS


class AutoMLTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        df = load_data(Path(__file__).resolve().parents[1] / 'data/battery_modeling_ready.parquet')
        cls.dev = df.loc[df.split.eq('train') & df.has_label].reset_index(drop=True)

    def test_group_isolation_inner_and_outer(self):
        for tr, va in GroupKFold(5).split(self.dev, groups=self.dev.group_id):
            a, b = self.dev.iloc[tr], self.dev.iloc[va]
            self.assertTrue(set(a.group_id).isdisjoint(b.group_id))
            for it, iv in GroupKFold(4).split(a, groups=a.group_id):
                self.assertTrue(set(a.iloc[it].group_id).isdisjoint(a.iloc[iv].group_id))

    def test_saved_predictor_uses_cycles_and_excludes_target(self):
        features = FEATURE_SETS['S1']
        learner = ScaledSVR(task='regression', C=1, gamma=.1, epsilon=.05)
        learner.fit(self.dev[features], np.log(self.dev.cycle_life))
        model = LifePredictor(learner, features)
        p = model.predict(self.dev)
        self.assertTrue(np.all(p > 100))
        changed = self.dev.copy()
        changed['cycle_life'] = 1
        np.testing.assert_allclose(p, model.predict(changed))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'model.joblib'
            joblib.dump(model, path)
            np.testing.assert_allclose(p, joblib.load(path).predict(self.dev))

    def test_custom_metric_original_units(self):
        class Predictor:
            def predict(self, x):
                return np.log([110, 180])
        score, details = original_mape(None, np.log([100, 200]), Predictor(), None, None, None)
        self.assertAlmostEqual(score, .1)
        self.assertAlmostEqual(details['MAPE_percent'], 10)


if __name__ == '__main__':
    unittest.main()
