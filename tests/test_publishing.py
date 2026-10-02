import json
import unittest
from pathlib import Path
from dotenv import dotenv_values
from scripts.make_public_bundle import public_paths, clean_notebook

ROOT = Path(__file__).resolve().parents[1]


class PublishingTests(unittest.TestCase):
    def test_git_ignore_excludes_private_artifacts(self):
        names = {p.relative_to(ROOT).as_posix() for p in public_paths(ROOT)}
        self.assertIn('.gitignore', names)
        self.assertIn('.env.example', names)
        self.assertIn('README.md', names)
        self.assertNotIn('.env', names)
        self.assertNotIn('data/battery_modeling_ready.parquet', names)
        self.assertNotIn('results/final_svr/final_model.joblib', names)
        self.assertNotIn('results/optuna/studies.sqlite3', names)
        self.assertNotIn('results/final_svr/predictions.csv', names)
        self.assertIn('results/final_svr/model_performance.csv', names)
        self.assertFalse(any('__pycache__' in name for name in names))

    def test_env_template_contains_only_data_path(self):
        self.assertEqual(dict(dotenv_values(ROOT/'.env.example')), {'BATTERY_DATA_DIR': 'data/raw'})

    def test_clean_notebook_preserves_source_without_outputs(self):
        raw = {'nbformat': 4, 'nbformat_minor': 5,
               'metadata': {'kernelspec': {'name':'python3'}, 'widgets': {'private':True}},
               'cells': [{'cell_type':'code', 'source':['print(1)'], 'metadata': {'private':True},
                          'execution_count':3, 'outputs':[{'output_type':'stream', 'text':'private'}]},
                         {'cell_type':'markdown', 'source':['# analysis'], 'metadata':{},
                          'attachments':{'private.png':{'image/png':'data'}}}]}
        clean = json.loads(clean_notebook(json.dumps(raw).encode()))
        self.assertEqual(clean['cells'][0]['source'], ['print(1)'])
        self.assertEqual(clean['cells'][0]['outputs'], [])
        self.assertIsNone(clean['cells'][0]['execution_count'])
        self.assertNotIn('widgets', clean['metadata'])
        self.assertNotIn('attachments', clean['cells'][1])


if __name__ == '__main__':
    unittest.main()
