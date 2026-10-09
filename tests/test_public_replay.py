"""Synthetic public-adapter checks; no model calls or source poem text."""
import copy
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import replay_results as replay


class PublicReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.analysis = replay.load_analysis(ROOT)

    def row(self, valid=True):
        row = dict(split='main300', model='synthetic', condition='S', replicate=1, id='x',
            reference_type='M', reference_features=['PURE'], predicted_type='M', predicted_features=['PURE'])
        row.update(self.analysis.score_prediction({'position':'M','features':['PURE']},
                   {'type':'M','features':['PURE']}, valid))
        return row

    def test_recompute_retains_invalid_in_denominator(self):
        rows = replay.recompute_rows([self.row(False)], self.analysis)
        self.assertEqual((len(rows), rows[0]['joint'], rows[0]['feature_fn']), (1, 0, 1))

    def test_forged_score_is_rejected(self):
        row = self.row();row['joint'] = 0
        with self.assertRaises(ValueError): replay.recompute_rows([row], self.analysis)

    def test_duplicate_prediction_is_rejected(self):
        with self.assertRaises(ValueError): replay.recompute_rows([self.row(), self.row()], self.analysis)

    def test_unknown_feature_is_rejected(self):
        row=self.row();row['predicted_features']=['INVENTED']
        with self.assertRaises(ValueError): replay.recompute_rows([row], self.analysis)

    def test_unknown_type_cannot_be_valid(self):
        row=self.row();row['predicted_type']='UNKNOWN'
        with self.assertRaises(ValueError): replay.recompute_rows([row], self.analysis)

    def test_exact_comparison_rejects_missing_and_changed_fields(self):
        for other in ({}, {'a':0.5000000001}, {'a':'0.5'}):
            with self.assertRaises(ValueError): replay.compare({'a':0.5}, other)


if __name__ == '__main__': unittest.main()
