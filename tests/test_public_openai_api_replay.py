"""Synthetic public replay checks; no study labels, evidence, or inference."""
from __future__ import annotations
import copy
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import random
import shutil
import sys
import unittest
from unittest.mock import patch
import uuid

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT if (ROOT/'tools/replay_results.py').is_file() else ROOT/'publication/greek-rhyme-eval-github'
ADAPTER = ROOT/'tools/replay_openai_api_results.py'
if not ADAPTER.is_file():
    ADAPTER = ROOT/'tmp/replay_openai_api_results.py'


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextmanager
def synthetic_public_root():
    # Windows CPython 3.12's mode-0700 temporary directory ACL can exclude the
    # desktop sandbox. Use an ordinary isolated directory with inherited ACLs.
    parent=ROOT.resolve()
    path=parent/('synthetic_replay_'+uuid.uuid4().hex)
    path.mkdir()
    try:
        yield path
    finally:
        if path.parent.resolve()!=parent or not path.resolve().is_relative_to(parent) or path.is_symlink():
            raise ValueError('Synthetic cleanup target escaped the test workspace')
        shutil.rmtree(path)


class PublicOpenAIAPIReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter=load(ADAPTER,'synthetic_openai_public_adapter')
        cls.previous=load(PUBLIC/'tools/replay_extension_results.py','synthetic_previous_adapter')
        cls.base=load(PUBLIC/'tools/replay_results.py','synthetic_base_adapter')
        cls.analysis=load(ROOT/cls.adapter.EXTENSION/'analyze.py','synthetic_api_analysis')
        cls.comparator=load(ROOT/cls.adapter.EXTENSION/'compare_codex.py','synthetic_api_comparator')
        cls.old=cls.analysis.OLD

    def fixture(self,full=False,failed=True):
        a=self.adapter
        count=300 if full else 10
        models=a.MODELS if full else {'astra_api':a.MODELS['astra_api']}
        reference={f'artificial_{i:03d}':{'id':f'artificial_{i:03d}','batch':(i-1)//10+1,
            'group':f'group_{i%7:03d}','reference_type':('M','F2','F3')[i%3],
            'reference_features':(['PURE'],['RICH'],[])[i%3]} for i in range(1,count+1)}
        protocol={'models':models.copy(),'main_pairs':count,'replicates':2,'batch_size':10,
            'conditions':dict.fromkeys(a.CONDITIONS,'synthetic condition'),
            'counts':{'main_prediction_records':count*len(models)*12,
                'total_planned_calls':count*len(models)*12//10},
            'statistics':{'resamples':12,'bootstrap_seed':20261008},
            'scope':['Artificial test observations only']}
        records,jobs=[],[]
        for model,requested in models.items():
            for condition in a.CONDITIONS:
                revision=condition in ('R','V','W')
                for rep in (1,2):
                    for batch in range(1,count//10+1):
                        initial_failed=failed and model=='astra_api' and rep==1 and batch==1
                        status=('blocked_by_initial_failure' if revision else 'transport_or_interrupted_failure') if initial_failed and (revision or condition=='S') else 'valid'
                        blocked=status=='blocked_by_initial_failure'
                        attempts=0 if blocked else 4 if initial_failed else 1
                        known=attempts if status=='valid' else 0
                        jid=f'{model}_{condition}_r{rep}_b{batch:02d}'
                        job=dict.fromkeys(a.JOB_KEYS)
                        job.update(job_id=jid,phase='phase2' if revision else 'phase1',split='main300',
                            model=model,model_requested=requested,condition=condition,replicate=rep,
                            batch=batch,items=10,status=status,
                            dependency=f'{model}_S_r{rep}_b{batch:02d}' if revision else None,
                            attempts=attempts,attempts_with_usage=known,attempts_reporting_cost=known,
                            all_attempt_seconds=attempts*.1 if known else None,
                            final_attempt_seconds=.1 if known else None,
                            incomplete_latency=bool(attempts and not known),incomplete_usage=known<attempts,
                            reported_cost_usd=attempts*.0003 if known else None,
                            input_tokens=attempts*25 if known else None,
                            output_tokens=attempts*12 if known else None,cost_is_complete=known==attempts)
                        jobs.append(job)
                        for ref in (r for r in reference.values() if r['batch']==batch):
                            valid=status=='valid'
                            pred={'type':ref['reference_type'],'features':ref['reference_features']} if valid else None
                            if valid and condition=='E' and int(ref['id'][-3:])%4==0:
                                pred={'type':'NONE','features':[]}
                            row={**ref,'split':'main300','model':model,'model_requested':requested,
                                'condition':condition,'replicate':rep,'job_id':jid,'status':status,
                                'predicted_type':pred['type'] if pred else None,
                                'predicted_features':pred['features'] if pred else []}
                            row.update(self.old.score_prediction({'position':ref['reference_type'],
                                'features':ref['reference_features']},pred,valid))
                            records.append(row)
        return records,jobs,protocol,reference

    def test_cumulative_attempts_and_own_initial_failure_keep_denominators(self):
        rows,jobs,protocol,reference=self.fixture()
        result=self.adapter.validate_design(rows,jobs,protocol,reference)
        self.assertEqual(result['prediction_records'],120)
        self.assertEqual(result['planned_job_records'],12)
        self.assertEqual(result['status_counts']['blocked_by_initial_failure'],3)
        self.assertTrue(any(j['attempts']==4 for j in jobs))

    def test_missing_and_duplicate_planned_rows_or_jobs_fail(self):
        for what in ('row','job','duplicate'):
            rows,jobs,protocol,reference=self.fixture()
            if what=='row':rows.pop()
            elif what=='job':jobs.pop()
            else:rows.append(copy.deepcopy(rows[0]))
            with self.subTest(what=what),self.assertRaises(ValueError):
                self.adapter.validate_design(rows,jobs,protocol,reference)

    def test_private_fields_and_unrecognized_status_fail(self):
        for field in ('historical_attempt_sources','selected_evidence','private_path'):
            rows,jobs,protocol,reference=self.fixture()
            jobs[0][field]='omitted'
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'aggregate job fields'):
                self.adapter.validate_design(rows,jobs,protocol,reference)
        rows,jobs,protocol,reference=self.fixture()
        rows[0]['response_sha256']='omitted'
        with self.assertRaisesRegex(ValueError,'prediction fields'):
            self.adapter.validate_design(rows,jobs,protocol,reference)
        rows,jobs,protocol,reference=self.fixture()
        rows[0]['status']='budget_deferred'
        with self.assertRaisesRegex(ValueError,'status'):
            self.adapter.validate_design(rows,jobs,protocol,reference)

    def test_reference_group_and_shared_dependency_are_fixed(self):
        for target,key,value in (('row','group','group_changed'),('row','reference_type','NONE'),
                                 ('job','dependency','another_model_S_r1_b01')):
            rows,jobs,protocol,reference=self.fixture()
            (rows[0] if target=='row' else next(j for j in jobs if j['condition']=='R'))[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                self.adapter.validate_design(rows,jobs,protocol,reference)

    def test_first_pass_cannot_be_blocked(self):
        rows,jobs,protocol,reference=self.fixture()
        job=next(j for j in jobs if j['condition']=='S' and j['replicate']==1)
        job.update(status='blocked_by_initial_failure',attempts=0,incomplete_latency=False,
                   incomplete_usage=False,cost_is_complete=True)
        for row in rows:
            if row['job_id']==job['job_id']:row['status']=job['status']
        with self.assertRaisesRegex(ValueError,'Only a dependent revision'):
            self.adapter.validate_design(rows,jobs,protocol,reference)

    def test_reporting_counts_and_completeness_are_coherent(self):
        for update in ({'attempts':True},{'attempts':-1},{'attempts':0},
                       {'attempts_reporting_cost':5},{'attempts_with_usage':.5},
                       {'cost_is_complete':True},{'incomplete_usage':False},
                       {'incomplete_latency':0},{'reported_cost_usd':float('nan')},
                       {'input_tokens':True}):
            rows,jobs,protocol,reference=self.fixture()
            jobs[0].update(update)
            with self.subTest(update=update),self.assertRaises(ValueError):
                self.adapter.validate_design(rows,jobs,protocol,reference)

    def test_blocked_jobs_cannot_claim_resources(self):
        for field in ('input_tokens','reported_cost_usd','all_attempt_seconds','attempts_reporting_cost'):
            rows,jobs,protocol,reference=self.fixture()
            next(j for j in jobs if j['status']=='blocked_by_initial_failure')[field]=1
            with self.subTest(field=field),self.assertRaises(ValueError):
                self.adapter.validate_design(rows,jobs,protocol,reference)

    def test_unknown_cost_is_not_silently_converted_to_measured_zero(self):
        rows,jobs,protocol,reference=self.fixture()
        jobs[0]['reported_cost_usd']=0.0
        with self.assertRaisesRegex(ValueError,'Missing reported cost'):
            self.adapter.validate_design(rows,jobs,protocol,reference)
        rows,jobs,protocol,reference=self.fixture()
        jobs[1]['reported_cost_usd']=None
        with self.assertRaisesRegex(ValueError,'Missing reported cost'):
            self.adapter.validate_design(rows,jobs,protocol,reference)

    def test_revision_cannot_run_after_its_own_initial_failure(self):
        rows,jobs,protocol,reference=self.fixture()
        job=next(j for j in jobs if j['status']=='blocked_by_initial_failure')
        job.update(status='transport_or_interrupted_failure',attempts=1,
                   incomplete_latency=True,incomplete_usage=True,cost_is_complete=False)
        for row in rows:
            if row['job_id']==job['job_id']:row['status']=job['status']
        with self.assertRaisesRegex(ValueError,'Blocked revision'):
            self.adapter.validate_design(rows,jobs,protocol,reference)

    def test_scores_and_exact_aggregate_comparisons_reject_tampering(self):
        rows,*_=self.fixture()
        rows[0]['joint']=1
        with self.assertRaises(ValueError):self.base.recompute_rows(rows,self.old)
        for value in (.100000000001,'0.1',None):
            with self.assertRaises(ValueError):self.base.compare({'v':.1},{'v':value})

    def test_recorded_arithmetic_is_selected_independently_of_host(self):
        values=[1e16,1.0,-1e16]
        self.assertEqual(self.adapter.arithmetic_for('3.11.9',self.previous)(values),0.0)
        self.assertEqual(self.adapter.arithmetic_for('3.12.14',self.previous)(values),1.0)
        rng=random.Random(20261011)
        if sys.version_info[:2] in ((3,11),(3,12)):
            fn=self.adapter.arithmetic_for('.'.join(map(str,sys.version_info[:3])),self.previous)
            for _ in range(20):
                sample=[rng.choice((1/3,.1,1e-8,-1.0,0,True,6)) for _ in range(80)]
                self.assertEqual(fn(sample),sum(sample))
        for value in ('3.13.0','3.12','unknown',None):
            with self.assertRaises(ValueError):self.adapter.arithmetic_for(value,self.previous)

    def test_final_provenance_and_both_runtime_records_are_required(self):
        good={'analysis_status':'complete_planned_denominators','missing_final_jobs':[],
              'runtime':{'analysis_python_version':'3.12.14','comparison_python_version':'3.12.14'}}
        self.assertEqual(self.adapter.require_final_runtime(good),good['runtime'])
        for key,value in (('analysis_status','partial'),('missing_final_jobs',['uncommitted']),('runtime',{})):
            bad=copy.deepcopy(good);bad[key]=value
            with self.assertRaises(ValueError):self.adapter.require_final_runtime(bad)

    def test_public_paths_reject_private_or_traversing_inputs(self):
        for value in ('../runs/meta.json','/absolute.json','C'+':/'+'private.json','a\\b','a//b','./a','',None):
            with self.subTest(value=value),self.assertRaises(ValueError):
                self.adapter.public_file(ROOT,value)

    def test_output_rejects_linked_build_parent_and_external_paths(self):
        with synthetic_public_root() as root:
            build=root/'build'
            build.mkdir()
            self.assertEqual(self.adapter.output_path(root),build/'openai_api_replay')
            for value in (root,build,root/'other',root.parent/'outside'):
                with self.subTest(value=value),self.assertRaises(ValueError):
                    self.adapter.output_path(root,value)
            original=Path.is_symlink
            with patch.object(Path,'is_symlink',lambda p: p==build or original(p)):
                with self.assertRaisesRegex(ValueError,'Linked public'):
                    self.adapter.output_path(root)
                with self.assertRaisesRegex(ValueError,'Linked public'):
                    self.adapter.public_file(root,'build/anything.json')

    def test_full_synthetic_numerics_and_matched_comparator(self):
        rows,jobs,protocol,reference=self.fixture(full=True)
        old_rows=copy.deepcopy(rows)
        old_jobs=copy.deepcopy(jobs)
        for row in old_rows+old_jobs:
            row['model']=row['model'].removesuffix('_api')
            row['model_requested']=row['model_requested'].removeprefix('openai/')
            row['job_id']=row['job_id'].replace('_api_','_')
            if 'dependency' in row and row['dependency']:
                row['dependency']=row['dependency'].replace('_api_','_')
        old_protocol={**protocol,'models':{k.removesuffix('_api'):v.removeprefix('openai/') for k,v in protocol['models'].items()}}
        self.analysis.sum=self.old.sum=self.previous.sum_312
        old_items=self.old.mean_replicates(old_rows,expected_replicates=2)
        old_summary=self.analysis.build_summary(old_rows,old_items,old_jobs,old_protocol,resamples=12,seed=20261008)
        # Independent expected collector semantics: frozen builder plus its
        # declared annotation, without using compute_results to construct truth.
        # On native 3.12 this also checks the complete arithmetic emulation.
        if sys.version_info[:2]==(3,12):
            del self.analysis.sum
            del self.old.sum
        expected_items=self.old.mean_replicates(rows,expected_replicates=2)
        expected_summary=self.analysis.build_summary(rows,expected_items,jobs,protocol,resamples=12,seed=20261008)
        expected_summary['metric_notes']['cost'] += (' Administrative continuation histories are retained; all prior attempts, '
            'including billing rejections with unmeasured cost, are included once.')
        expected_summary={k:expected_summary[k] for k in self.adapter.SUMMARY_KEYS}
        runtime={'analysis_python_version':'3.12.14','comparison_python_version':'3.12.14'}
        summary,items,comparison,counts=self.adapter.compute_results(rows,jobs,protocol,reference,
            old_summary,old_items,old_jobs,self.base,self.previous,self.analysis,self.comparator,runtime)
        self.base.compare(summary,expected_summary)
        self.base.compare(items,expected_items)
        self.assertEqual(counts['prediction_records'],10800)
        self.assertEqual(counts['planned_job_records'],1080)
        self.assertEqual(counts['cells'],18)
        self.assertEqual(len(items),5400)
        self.assertEqual(len(comparison['cells']),18)
        self.assertEqual(len(comparison['reliability']),6)
        self.assertEqual(summary['cost_total']['committed_attempts'],
                         sum(j['attempts'] for j in jobs))
        self.assertTrue(all(cell['paired_batch_differences']['joint']['estimate']==0 for cell in comparison['cells']))
        # Persist only fabricated records to a temporary public layout. This
        # exercises source hashes, sanitization, imports and the real replay CLI
        # function without reading any real cohort output.
        with synthetic_public_root() as directory:
            root=Path(directory)
            def save(relative,value):
                target=root/relative;target.parent.mkdir(parents=True,exist_ok=True)
                target.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
            sources={}
            for relative in self.adapter.SOURCE_FILES:
                target=root/relative;target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(ROOT/relative,target);sources[relative]=self.adapter.sha(target)
            for relative in ('tools/replay_results.py','tools/replay_extension_results.py','study_2026-10-08/analysis.py'):
                target=root/relative;target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(PUBLIC/relative,target)
            save('provenance/public_release.json',{'original_analysis_sha256':self.adapter.sha(root/'study_2026-10-08/analysis.py')})
            prefix='provenance/'+self.adapter.EXTENSION+'/'
            save(prefix+'public_release.json',{'source_code_sha256':sources,'runtime':runtime,
                'analysis_status':'complete_planned_denominators','missing_final_jobs':[]})
            save(prefix+'protocol.json',protocol)
            for name,value in (('summary',old_summary),('per_item',old_items),('per_prediction',old_rows),('jobs',old_jobs)):
                save('results/'+name+'.json',value)
            prefix='results/'+self.adapter.EXTENSION+'/'
            for name,value in (('summary',expected_summary),('per_item',expected_items),('per_prediction',rows),('jobs',jobs),('comparison_codex_matched',comparison)):
                save(prefix+name+'.json',value)
            def manifest():
                hashes={p.relative_to(root).as_posix():self.adapter.sha(p) for p in root.rglob('*')
                    if p.is_file() and p.relative_to(root).as_posix()!='provenance/public_manifest.json'}
                save('provenance/public_manifest.json',{'sha256':hashes})
            manifest()
            actual,matched,report=self.adapter.replay(root)
            self.base.compare(actual,summary);self.base.compare(matched,comparison)
            self.assertEqual(report['status'],'PASS')
            self.assertEqual(report['new_model_calls'],0)
            tampered=copy.deepcopy(comparison)
            tampered['cells'][0]['paired_batch_differences']['joint']['estimate']=.001
            save(prefix+'comparison_codex_matched.json',tampered);manifest()
            with self.assertRaisesRegex(ValueError,'Result value differs'):
                self.adapter.replay(root)
        # Comparator itself rejects changed source grouping before statistics.
        old_items[0]['group']='group_changed'
        with self.assertRaisesRegex(ValueError,'item contract differs'):
            self.comparator.build_comparison(self.adapter.complete_view(summary),items,jobs,
                self.adapter.complete_view(old_summary),old_items,old_jobs,self.old,12,20261008)


if __name__=='__main__':
    unittest.main()
