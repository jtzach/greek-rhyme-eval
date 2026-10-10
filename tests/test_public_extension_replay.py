"""Synthetic extension replay checks; no inference or source poem text."""
import copy
import importlib.util
import random
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
ADAPTER_PATH = ROOT / 'tools/replay_extension_results.py'


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class PublicExtensionReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = load(ADAPTER_PATH, 'public_extension_adapter_under_test')
        cls.base = load(ROOT / 'tools/replay_results.py', 'public_base_adapter_for_extension_tests')
        cls.old = cls.base.load_analysis(ROOT)

    def fixture(self, failed=False):
        adapter, old = self.adapter, self.old
        reference = {
            'synthetic_1':{'id':'synthetic_1','batch':1,'group':'group_001',
                           'reference_type':'M','reference_features':['PURE']},
            'synthetic_2':{'id':'synthetic_2','batch':1,'group':'group_002',
                           'reference_type':'F2','reference_features':['RICH']}}
        protocol = {'models':{'synthetic':'synthetic/model'}, 'main_pairs':2,
            'batch_size':2,'replicates':2,'conditions':dict.fromkeys(adapter.CONDITIONS,'synthetic'),
            'counts':{'main_prediction_records':24,'total_planned_calls':12}}
        records, jobs = [], []
        for condition in adapter.CONDITIONS:
            for replicate in (1,2):
                revision = condition in ('R','V','W')
                status = ('blocked_by_initial_failure' if revision else 'invalid_format') if failed and replicate == 1 and (revision or condition == 'S') else 'valid'
                jid = f'synthetic_{condition}_r{replicate}_b01'
                job = dict.fromkeys(adapter.JOB_KEYS)
                job.update(job_id=jid,phase='phase2' if revision else 'phase1',split='main300',
                    model='synthetic',model_requested='synthetic/model',condition=condition,
                    replicate=replicate,batch=1,items=2,status=status,
                    dependency=f'synthetic_S_r{replicate}_b01' if revision else None,
                    attempts=0 if status == 'blocked_by_initial_failure' else 1,
                    attempts_with_usage=0 if status == 'blocked_by_initial_failure' else 1,
                    all_attempt_seconds=None if status == 'blocked_by_initial_failure' else 1.5,
                    final_attempt_seconds=None if status == 'blocked_by_initial_failure' else 1.5,
                    incomplete_latency=False,incomplete_usage=False,
                    reported_cost_usd=None if status == 'blocked_by_initial_failure' else .001,
                    attempts_reporting_cost=0 if status == 'blocked_by_initial_failure' else 1,
                    cost_is_complete=True)
                jobs.append(job)
                for item in reference.values():
                    valid = status == 'valid'
                    prediction = {'type':item['reference_type'],'features':item['reference_features']} if valid else None
                    row = {**item,'split':'main300','model':'synthetic','model_requested':'synthetic/model',
                        'condition':condition,'replicate':replicate,'job_id':jid,'status':status,
                        'predicted_type':prediction['type'] if prediction else None,
                        'predicted_features':prediction['features'] if prediction else []}
                    row.update(old.score_prediction({'position':item['reference_type'],'features':item['reference_features']},prediction,valid))
                    records.append(row)
        return records,jobs,protocol,reference

    def test_extension_sum_preserves_compensation_and_integer_types(self):
        total=self.adapter.sum_312
        self.assertEqual(total([1e16,1.0,-1e16]),1.0)
        self.assertEqual(total([1e16,1,-1e16]),0.0)
        self.assertEqual(total([1.0,1e100,1.0,-1e100]),2.0)
        self.assertEqual(total([0.1]*10),1.0)
        self.assertEqual(total([True,False,2]),3)
        self.assertIs(type(total([True,False,2])),int)
        self.assertIs(type(total([])),int)
        self.assertIs(type(total([1.0,2.0])),float)

    def test_extension_sum_rejects_unsupported_numeric_domain(self):
        for values in ([float('nan')],[float('inf')],['1'],[1+0j],[1.0,2**32],[-1,2**31,1e16,1.0,-1e16],[2**100,-2**100,1e16,1.0,-1e16]):
            with self.subTest(values=values),self.assertRaises(ValueError):
                self.adapter.sum_312(values)
        for start in (True,0.0,1,None):
            with self.subTest(start=start),self.assertRaises(ValueError):
                self.adapter.sum_312([],start)

    def test_extension_sum_matches_native_python312(self):
        if sys.version_info[:2] != (3,12):
            self.skipTest('Native CPython 3.12 cross-check; fixed vectors run on every supported host')
        rng=random.Random(20261010)
        cases=[[],[True,False,4],[1e16,1.0,-1e16],[1e16,1,-1e16]]
        cases += [[rng.choice((0.1,1/3,1e-8,-1.0,10**6,0,True)) for _ in range(80)] for _ in range(100)]
        for values in cases:
            self.assertEqual(self.adapter.sum_312(values),sum(values))
            self.assertIs(type(self.adapter.sum_312(values)),type(sum(values)))

    def test_complete_design_and_blocked_failures_retain_denominators(self):
        for failed in (False,True):
            records,jobs,protocol,reference = self.fixture(failed)
            report = self.adapter.validate_design(records,jobs,protocol,reference)
            self.assertEqual(report['prediction_records'],24)
            self.assertEqual(report['planned_job_records'],12)
            self.assertEqual(len(self.base.recompute_rows(records,self.old)),24)
        self.assertEqual(report['status_counts']['blocked_by_initial_failure'],3)

    def test_missing_prediction_or_job_is_rejected(self):
        for remove_prediction in (False,True):
            records,jobs,protocol,reference = self.fixture(True)
            if remove_prediction:
                records.pop()
            else:
                jobs.pop()
            with self.assertRaises(ValueError):
                self.adapter.validate_design(records,jobs,protocol,reference)

    def test_duplicate_prediction_is_rejected(self):
        records,jobs,protocol,reference = self.fixture()
        records.append(copy.deepcopy(records[0]))
        with self.assertRaisesRegex(ValueError,'duplicate'):
            self.adapter.validate_design(records,jobs,protocol,reference)

    def test_reference_batch_and_group_changes_are_rejected(self):
        for field,value in (('reference_type','F3'),('reference_features',[]),('batch',2),('group','group_003')):
            records,jobs,protocol,reference = self.fixture()
            records[0][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError,'Reference, batch or opaque'):
                self.adapter.validate_design(records,jobs,protocol,reference)

    def test_validity_mismatch_and_raw_response_hash_are_rejected(self):
        records,jobs,protocol,reference = self.fixture()
        records[0]['valid'] = 0
        with self.assertRaisesRegex(ValueError,'validity'):
            self.adapter.validate_design(records,jobs,protocol,reference)
        records,jobs,protocol,reference = self.fixture()
        records[0]['response_sha256'] = 'forbidden-public-field'
        with self.assertRaisesRegex(ValueError,'Unexpected public prediction fields'):
            self.adapter.validate_design(records,jobs,protocol,reference)

    def test_forged_scores_and_unknown_features_are_rejected(self):
        records,_,_,_ = self.fixture()
        records[0]['joint'] = 0
        with self.assertRaises(ValueError):
            self.base.recompute_rows(records,self.old)
        records,_,_,_ = self.fixture()
        records[0]['predicted_features'] = ['UNKNOWN_FEATURE']
        with self.assertRaises(ValueError):
            self.base.recompute_rows(records,self.old)

    def test_revision_cannot_survive_a_failed_initial_answer(self):
        records,jobs,protocol,reference = self.fixture(True)
        revision = next(j for j in jobs if j['condition']=='R' and j['replicate']==1)
        revision.update(status='invalid_format',attempts=1,incomplete_usage=True,
                        incomplete_latency=True,cost_is_complete=False)
        for row in records:
            if row['job_id']==revision['job_id']:
                row['status']='invalid_format'
        with self.assertRaisesRegex(ValueError,'Blocked revision'):
            self.adapter.validate_design(records,jobs,protocol,reference)

    def test_wrong_dependency_and_nonfinite_cost_rejected(self):
        records,jobs,protocol,reference = self.fixture()
        next(j for j in jobs if j['condition']=='R')['dependency']='other-job'
        with self.assertRaisesRegex(ValueError,'dependency'):
            self.adapter.validate_design(records,jobs,protocol,reference)
        records,jobs,protocol,reference = self.fixture()
        jobs[0]['reported_cost_usd']=float('nan')
        with self.assertRaisesRegex(ValueError,'aggregate'):
            self.adapter.validate_design(records,jobs,protocol,reference)

    def test_reporting_counts_require_bounded_integers(self):
        for field in ('attempts_with_usage', 'attempts_reporting_cost'):
            for value in (-1, 99, .5, True, None):
                records,jobs,protocol,reference = self.fixture()
                jobs[0][field] = value
                with self.subTest(field=field,value=value), self.assertRaisesRegex(ValueError,'reporting count'):
                    self.adapter.validate_design(records,jobs,protocol,reference)
        records,jobs,protocol,reference = self.fixture()
        jobs[0]['attempts'] = True
        with self.assertRaisesRegex(ValueError,'Attempt count'):
            self.adapter.validate_design(records,jobs,protocol,reference)

    def test_completeness_flags_require_booleans_and_coherent_counts(self):
        for field in ('incomplete_latency', 'incomplete_usage', 'cost_is_complete'):
            for value in (0, 1, None, 'true'):
                records,jobs,protocol,reference = self.fixture()
                jobs[0][field] = value
                with self.subTest(field=field,value=value), self.assertRaisesRegex(ValueError,'flags'):
                    self.adapter.validate_design(records,jobs,protocol,reference)
        for update in ({'attempts_with_usage':0}, {'incomplete_usage':True},
                       {'attempts_reporting_cost':0}, {'cost_is_complete':False}):
            records,jobs,protocol,reference = self.fixture()
            jobs[0].update(update)
            with self.subTest(update=update), self.assertRaisesRegex(ValueError,'completeness'):
                self.adapter.validate_design(records,jobs,protocol,reference)

    def test_blocked_jobs_cannot_report_unincurred_work(self):
        for field in ('reported_cost_usd', 'input_tokens', 'cached_input_tokens',
                      'cache_write_input_tokens', 'output_tokens', 'reasoning_output_tokens',
                      'all_attempt_seconds', 'final_attempt_seconds', 'attempts_with_usage',
                      'attempts_reporting_cost', 'incomplete_latency'):
            records,jobs,protocol,reference = self.fixture(True)
            blocked = next(j for j in jobs if j['status']=='blocked_by_initial_failure')
            blocked[field] = True if field == 'incomplete_latency' else 7
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.adapter.validate_design(records,jobs,protocol,reference)

    def test_unknown_failed_attempt_can_retain_missing_cost_usage_and_latency(self):
        records,jobs,protocol,reference = self.fixture()
        failed = next(j for j in jobs if j['condition']=='C' and j['replicate']==1)
        failed.update(status='transport_or_interrupted_failure',attempts_with_usage=0,
            attempts_reporting_cost=0,reported_cost_usd=None,incomplete_usage=True,
            incomplete_latency=True,cost_is_complete=False,all_attempt_seconds=None,
            final_attempt_seconds=None)
        for row in records:
            if row['job_id']==failed['job_id']:
                row.update(status=failed['status'],predicted_type=None,predicted_features=[])
                row.update(self.old.score_prediction({'position':row['reference_type'],
                    'features':row['reference_features']},None,False))
        self.adapter.validate_design(records,jobs,protocol,reference)

    def test_original_reference_contract_rejects_inconsistent_membership(self):
        records,_,_,_ = self.fixture()
        records[-1]['group']='group_099'
        with self.assertRaisesRegex(ValueError,'inconsistent'):
            self.adapter.reference_contract(records)

    def test_exact_comparison_rejects_missing_and_changed_fields(self):
        for other in ({},{'cost':.1000000001},{'cost':'0.1'}):
            with self.assertRaises(ValueError):
                self.base.compare({'cost':.1},other)

    def test_original_comparison_checks_group_alignment_before_statistics(self):
        records,_,_,_ = self.fixture()
        items = self.old.mean_replicates(records,expected_replicates=2)
        original = copy.deepcopy(items)
        next(r for r in original if r['condition']=='S')['group']='group_099'
        fake = SimpleNamespace(paired_contrast=lambda *args: self.fail('Statistics ran before reference validation'))
        with self.assertRaisesRegex(ValueError,'reference/batch/group'):
            self.adapter.compare_original_public({'cells':[]},items,{'cells':[]},original,fake,10,1)


if __name__ == '__main__':
    unittest.main()
