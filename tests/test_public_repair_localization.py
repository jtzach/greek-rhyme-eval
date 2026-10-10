"""Synthetic post hoc replay checks; no poem text, raw feedback or API calls."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/replay_repair_localization.py'


class RepairLocalizationReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('repair_localization_under_test', SCRIPT)
        cls.analysis = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.analysis)

    def fixture(self):
        ids = [f'pair_{i:03d}' for i in range(1,301)]
        groups = {self.analysis.CHANGED:ids[:50],self.analysis.UNCHANGED:ids[50:]}
        rows = [dict(id=identifier,model=model,model_requested='synthetic/'+model,
            split='main300',condition=condition,replicate=replicate,valid=1,
            predicted_type='M',predicted_features=['PURE'],reference_type='M',
            reference_features=['PURE'],joint=1,core_joint=1)
            for model in self.analysis.MODELS for condition in ('V','W')
            for replicate in (1,2) for identifier in ids]
        return rows,groups

    def test_output_changes_and_validity_transitions_are_distinct(self):
        rows,groups = self.fixture()
        def row(identifier,condition):
            return next(r for r in rows if r['model']=='astra' and r['replicate']==1
                and r['id']==identifier and r['condition']==condition)
        row('pair_001','W').update(predicted_type='F2',joint=0,core_joint=0)
        for identifier,conditions in [('pair_051',('V',)),('pair_052',('W',)),('pair_053',('V','W'))]:
            for condition in conditions:
                row(identifier,condition).update(valid=0,predicted_type=None,predicted_features=[],joint=0,core_joint=0)
        result = self.analysis.recompute(rows,groups)[0]['groups']
        changed,unchanged,overall = (result[k] for k in (self.analysis.CHANGED,self.analysis.UNCHANGED,'all'))
        self.assertEqual((changed['both_valid_output_changes'],changed['both_valid_joint_losses'],changed['net_joint_matches']),(1,1,-1))
        self.assertEqual((unchanged['validity_gains'],unchanged['validity_losses'],unchanged['both_invalid']),(1,1,1))
        self.assertEqual((unchanged['both_valid_output_changes'],unchanged['validity_or_label_changes'],unchanged['net_joint_matches']),(0,2,0))
        self.assertEqual((overall['paired_predictions'],overall['net_joint_matches']),(600,-1))
        self.assertAlmostEqual(overall['joint_delta_pp'],changed['contribution_to_overall_joint_delta_pp']+unchanged['contribution_to_overall_joint_delta_pp'])

    def test_missing_or_duplicate_planned_comparison_is_rejected(self):
        for duplicate in (False,True):
            rows,groups = self.fixture()
            rows.append(copy.deepcopy(rows[0])) if duplicate else rows.pop()
            with self.subTest(duplicate=duplicate),self.assertRaises(ValueError):
                self.analysis.recompute(rows,groups)

    def test_group_overlap_and_omitted_ids_are_rejected(self):
        for overlap in (False,True):
            rows,groups = self.fixture()
            if overlap:
                groups[self.analysis.UNCHANGED][0]=groups[self.analysis.CHANGED][0]
            else:
                groups[self.analysis.UNCHANGED].pop()
            with self.subTest(overlap=overlap),self.assertRaises(ValueError):
                self.analysis.recompute(rows,groups)

    def test_forged_joint_or_core_score_is_rejected(self):
        for key in ('joint','core_joint'):
            rows,groups = self.fixture()
            rows[0][key]=0
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'categorical labels'):
                self.analysis.recompute(rows,groups)

    def test_references_cannot_change_across_conditions(self):
        rows,groups = self.fixture()
        rows[-1].update(reference_type='F2',joint=0,core_joint=0)
        with self.assertRaisesRegex(ValueError,'References differ'):
            self.analysis.recompute(rows,groups)

    def test_public_replay_uses_only_public_files_and_rejects_changed_results(self):
        rows,groups = self.fixture()
        artifact={'analysis_type':'post_hoc_descriptive_repair_localization','groups':groups,
            'models':self.analysis.recompute(rows,groups),'analysis_code_sha256':self.analysis.sha(SCRIPT)}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            out=root/'results/extension_2026-10-10'
            out.mkdir(parents=True)
            (root/'results/per_prediction.json').write_text(json.dumps(rows),encoding='utf-8')
            (out/'per_prediction.json').write_text('[]',encoding='utf-8')
            target=out/'posthoc_repair_localization.json'
            target.write_text(json.dumps(artifact),encoding='utf-8')
            original_read=self.analysis.read
            opened=[]
            def only_public(path):
                resolved=Path(path).resolve()
                self.assertTrue(resolved.is_relative_to(root.resolve()))
                opened.append(resolved.relative_to(root.resolve()).as_posix())
                return original_read(path)
            self.analysis.read=only_public
            try:
                self.assertEqual(self.analysis.verify_public(root)['status'],'verified')
                self.assertEqual(set(opened),{'results/per_prediction.json','results/extension_2026-10-10/per_prediction.json','results/extension_2026-10-10/posthoc_repair_localization.json'})
                artifact['models'][0]['groups']['all']['joint_gains']=1
                target.write_text(json.dumps(artifact),encoding='utf-8')
                with self.assertRaisesRegex(ValueError,'replay differs'):
                    self.analysis.verify_public(root)
                artifact['models']=self.analysis.recompute(rows,groups)
                artifact['analysis_code_sha256']='0'*64
                target.write_text(json.dumps(artifact),encoding='utf-8')
                with self.assertRaisesRegex(ValueError,'code differs'):
                    self.analysis.verify_public(root)
            finally:
                self.analysis.read=original_read


if __name__ == '__main__':
    unittest.main()
