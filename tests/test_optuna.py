import unittest
from pathlib import Path
import numpy as np
from src.preprocess import load_data
from src.tune_optuna import build, FS


class OptunaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        df = load_data(Path(__file__).resolve().parents[1] / 'data/battery_modeling_ready.parquet')
        cls.dev = df.loc[df.split.eq('train')].copy()

    def test_feature_and_target_isolation(self):
        for fs, columns in FS.items():
            self.assertNotIn('cycle_life', columns)
            self.assertNotIn('outer_cv_fold', columns)
        for target in ['log', 'original']:
            params = {'feature_set': 'S4_early', 'target_transform': target,
                      'kernel': 'rbf', 'C': 1., 'gamma': .1,
                      'epsilon_log' if target == 'log' else 'epsilon_original': .05 if target == 'log' else 5.}
            model = build('SVR', params)
            model.fit(self.dev, self.dev.cycle_life)
            expected = model.predict(self.dev)
            changed = self.dev.copy()
            changed['cycle_life'] = 1
            np.testing.assert_allclose(expected, model.predict(changed))
            self.assertTrue(np.isfinite(expected).all())
            self.assertGreater(expected.mean(), 100)

    def test_imputer_is_fold_local(self):
        params = {'feature_set': 'S1', 'target_transform': 'log', 'kernel': 'rbf',
                  'C': 1., 'gamma': .1, 'epsilon_log': .05}
        frame = self.dev.iloc[:20].copy()
        frame.iloc[0, frame.columns.get_loc('DQ_relative_logvar')] = np.nan
        model = build('SVR', params)
        model.fit(frame, frame.cycle_life)
        stats = model.named_steps['features'].named_transformers_['early'].named_steps['imputer'].statistics_
        self.assertAlmostEqual(stats[0], frame.DQ_relative_logvar.median())


if __name__ == '__main__':
    unittest.main()
