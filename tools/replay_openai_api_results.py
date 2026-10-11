"""Offline exact replay of three API configurations paired with archived Codex.

No inference, raw-response extraction, private input access, or frozen CLIs.
Aggregate historical attempts do not individually attest dispatch or billing.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path, PurePosixPath
import re
import stat
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
EXTENSION = 'extension_2026-10-11_openai_api'
CONDITIONS = ('S','C','E','R','V','W')
MODELS = {'astra_api':'openai/gpt-6-astra', 'sol_api':'openai/gpt-6.1-sol',
          'luna_api':'openai/gpt-6-luna'}
SUMMARY_KEYS = ('reference_status','counts','statistics','metric_notes','scope',
                'cells','contrasts','cost_total')
COMPARISON_KEYS = ('status','cells','reliability','statistics','scope')
METRIC_KEYS = ('valid','stress','feature_exact','joint','feature_item_f1',
    'core_joint','feature_tp','feature_fp','feature_fn','none_with_nonempty_features','predicted_none')
PREDICTION_KEYS = ('id','split','model','model_requested','condition','replicate',
    'batch','group','job_id','status','reference_type','reference_features',
    'predicted_type','predicted_features') + METRIC_KEYS
JOB_KEYS = ('job_id','phase','split','model','model_requested','condition',
    'replicate','batch','items','status','dependency','attempts','attempts_with_usage',
    'all_attempt_seconds','final_attempt_seconds','incomplete_latency','incomplete_usage',
    'input_tokens','cached_input_tokens','cache_write_input_tokens','output_tokens',
    'reasoning_output_tokens','reported_cost_usd','attempts_reporting_cost','cost_is_complete')
STATUSES = {'valid','invalid_format','transport_or_interrupted_failure',
            'tool_use','blocked_by_initial_failure'}
SOURCE_FILES = tuple(EXTENSION+'/'+name for name in
    ('analyze.py','prepare.py','runner.py','compare_codex.py'))
CONTINUATION_COST_NOTE = (' Administrative continuation histories are retained; all prior attempts, '
    'including billing rejections with unmeasured cost, are included once.')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reject_link_components(root,path):
    try:
        parts=path.relative_to(root).parts
    except ValueError:
        raise ValueError('Artifact path must remain inside the public root') from None
    current=root
    for part in parts:
        current=current/part
        if current.is_symlink():
            raise ValueError('Linked public artifact paths are forbidden')
        try:
            # Junctions are directory links on Windows; Path.is_symlink alone
            # does not recognize them on the documented Python 3.11 runtime.
            tag=getattr(current.lstat(),'st_reparse_tag',None)
        except FileNotFoundError:
            continue
        if tag is not None and tag==getattr(stat,'IO_REPARSE_TAG_MOUNT_POINT',object()):
            raise ValueError('Junction public artifact paths are forbidden')


def public_file(root,relative):
    if (not isinstance(relative,str) or not relative or PurePosixPath(relative).is_absolute()
            or '\\' in relative or ':' in relative
            or any(part in ('','..','.') for part in relative.split('/'))):
        raise ValueError('Unsafe public artifact path')
    path=root/relative
    reject_link_components(root,path)
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError('Public artifact missing or outside release')
    return path


def sum_311(values,start=0):
    """Default-zero finite scalar domain of the recorded 3.11 analysis."""
    if type(start) is not int or start != 0:
        raise ValueError('Replay sum supports the recorded default-zero start only')
    total=start
    for value in values:
        if type(value) not in (int,float,bool) or (type(value) is float and not math.isfinite(value)):
            raise ValueError('Replay sums require finite built-in numeric values')
        total=total+value
    return total


def arithmetic_for(version,previous_adapter):
    if not isinstance(version,str) or not re.fullmatch(r'3\.(?:11|12)\.[0-9]+',version):
        raise ValueError('Recorded runtime has no validated replay arithmetic')
    return sum_311 if version.startswith('3.11.') else previous_adapter.sum_312


def require_final_runtime(provenance):
    if (provenance.get('analysis_status') != 'complete_planned_denominators'
            or provenance.get('missing_final_jobs') != []):
        raise ValueError('Public provenance must describe a completed final analysis')
    runtime=provenance.get('runtime')
    if not isinstance(runtime,dict) or set(runtime) != {'analysis_python_version','comparison_python_version'}:
        raise ValueError('Both recorded analysis and comparison runtimes are required')
    return runtime


def validate_design(records, jobs, protocol, reference):
    """Reject omitted failures, extra rows, changed cells, and mismatched lineage."""
    models = protocol['models']
    if len(reference) != protocol['main_pairs'] or protocol['replicates'] != 2:
        raise ValueError('Reference cardinality or repetition design differs')
    if tuple(protocol['conditions']) != CONDITIONS:
        raise ValueError('Conditions differ from the frozen six-condition design')
    batch_ids = sorted({r['batch'] for r in reference.values()})
    per_batch = Counter(r['batch'] for r in reference.values())
    if any(n != protocol['batch_size'] for n in per_batch.values()):
        raise ValueError('Reference batch size differs')
    expected = set(itertools.product(models, CONDITIONS, (1, 2), reference))
    observed = set()
    by_job = {}
    for row in records:
        if set(row) != set(PREDICTION_KEYS):
            raise ValueError('Unexpected public prediction fields')
        key = row['model'], row['condition'], row['replicate'], row['id']
        if key not in expected or key in observed:
            raise ValueError('Unexpected or duplicate planned prediction')
        observed.add(key)
        if row['split'] != 'main300' or row['model_requested'] != models[row['model']]:
            raise ValueError('Prediction model or split differs')
        if any(row[k] != value for k, value in reference[row['id']].items()):
            raise ValueError('Reference, batch or opaque source group differs')
        jid = f"{row['model']}_{row['condition']}_r{row['replicate']}_b{row['batch']:02d}"
        if row['job_id'] != jid or row['status'] not in STATUSES:
            raise ValueError('Prediction job identity or status differs')
        if bool(row['valid']) != (row['status'] == 'valid'):
            raise ValueError('Prediction validity and status disagree')
        if not row['valid'] and (row['predicted_type'] is not None or row['predicted_features'] != []):
            raise ValueError('Invalid prediction must retain the failure encoding')
        by_job.setdefault(jid, []).append(row)
    if observed != expected or len(records) != protocol['counts']['main_prediction_records']:
        raise ValueError('Missing planned prediction records')
    expected_jobs = {f'{model}_{condition}_r{replicate}_b{batch:02d}'
                     for model, condition, replicate, batch in itertools.product(models, CONDITIONS, (1, 2), batch_ids)}
    job_map = {}
    for job in jobs:
        if set(job) != set(JOB_KEYS):
            raise ValueError('Unexpected public aggregate job fields')
        jid = job['job_id']
        if jid not in expected_jobs or jid in job_map:
            raise ValueError('Unexpected or duplicate planned job')
        rows = by_job[jid]
        first = rows[0]
        if any(job[k] != first[k] for k in ('split', 'model', 'model_requested', 'condition', 'replicate', 'batch', 'status')):
            raise ValueError('Job and prediction identities disagree')
        if len(rows) != job['items'] or len({r['status'] for r in rows}) != 1:
            raise ValueError('Batch status or item count differs')
        revision = job['condition'] in ('R', 'V', 'W')
        dependency = f"{job['model']}_S_r{job['replicate']}_b{job['batch']:02d}" if revision else None
        if job['dependency'] != dependency or job['phase'] != ('phase2' if revision else 'phase1'):
            raise ValueError('Shared S dependency or phase differs')
        blocked = job['status'] == 'blocked_by_initial_failure'
        if blocked and not revision:
            raise ValueError('Only a dependent revision can be blocked by its initial answer')
        if type(job['attempts']) is not int or (job['attempts'] != 0 if blocked else job['attempts'] < 1):
            raise ValueError('Attempt count differs from cumulative committed-history policy')
        for field in ('attempts_with_usage', 'attempts_reporting_cost'):
            if type(job[field]) is not int or not 0 <= job[field] <= job['attempts']:
                raise ValueError('Aggregate reporting count exceeds the integer attempt bounds')
        for field in ('incomplete_latency', 'incomplete_usage', 'cost_is_complete'):
            if type(job[field]) is not bool:
                raise ValueError('Aggregate completeness flags must be booleans')
        if job['cost_is_complete'] != (job['attempts_reporting_cost'] == job['attempts']):
            raise ValueError('Aggregate cost completeness disagrees with reporting count')
        if job['incomplete_usage'] != (job['attempts_with_usage'] < job['attempts']):
            raise ValueError('Aggregate usage completeness disagrees with reporting count')
        if (job['reported_cost_usd'] is None) != (job['attempts_reporting_cost'] == 0):
            raise ValueError('Missing reported cost disagrees with its reporting count')
        for field in JOB_KEYS[11:]:
            value = job[field]
            flag = field in ('incomplete_latency', 'incomplete_usage', 'cost_is_complete')
            if value is not None and (type(value) not in ((bool,) if flag else (int, float)) or not math.isfinite(value) or value < 0):
                raise ValueError('Invalid aggregate usage, latency or cost field')
        if not job['attempts']:
            empty_fields = ('all_attempt_seconds', 'final_attempt_seconds', 'input_tokens',
                'cached_input_tokens', 'cache_write_input_tokens', 'output_tokens',
                'reasoning_output_tokens', 'reported_cost_usd')
            if any(job[field] not in (None, 0) for field in empty_fields) or job['incomplete_latency']:
                raise ValueError('Blocked jobs cannot report cost, tokens or latency without attempts')
        job_map[jid] = job
    if set(job_map) != expected_jobs or len(jobs) != protocol['counts']['total_planned_calls']:
        raise ValueError('Missing planned job records')
    for job in jobs:
        if job['dependency'] is not None:
            should_block = job_map[job['dependency']]['status'] != 'valid'
            if (job['status'] == 'blocked_by_initial_failure') != should_block:
                raise ValueError('Blocked revision disagrees with its shared S outcome')
    return {'prediction_records': len(records), 'item_condition_records': len(records) // 2,
            'planned_job_records': len(jobs), 'cells': len(models) * len(CONDITIONS),
            'status_counts': dict(Counter(j['status'] for j in jobs))}


def complete_view(summary):
    # Sanitization intentionally drops execution metadata; final release gates
    # establish completeness, while replay independently checks every denominator.
    return {**summary,'analysis_status':'complete_planned_denominators','missing_final_jobs':[]}


def compute_results(records,jobs,protocol,reference,original_summary,original_items,
                    original_jobs,base,previous,analysis,comparator,runtime):
    """Pure numerical reconstruction; its arguments contain only public data."""
    analysis.sum=analysis.OLD.sum=arithmetic_for(runtime['analysis_python_version'],previous)
    rows=base.recompute_rows(records,analysis.OLD)
    counts=validate_design(rows,jobs,protocol,reference)
    items=analysis.OLD.mean_replicates(rows,expected_replicates=2)
    statistics=protocol['statistics']
    computed=analysis.build_summary(rows,items,jobs,protocol,
        resamples=statistics['resamples'],seed=statistics['bootstrap_seed'])
    # Exact nonnumeric annotation appended by the authorized collection adapter;
    # every scientific value still comes from unchanged frozen build_summary.
    computed['metric_notes']['cost'] += CONTINUATION_COST_NOTE
    computed={key:computed[key] for key in SUMMARY_KEYS}
    analysis.OLD.sum=comparator.sum=arithmetic_for(runtime['comparison_python_version'],previous)
    comparison=comparator.build_comparison(complete_view(computed),items,jobs,
        complete_view(original_summary),original_items,original_jobs,analysis.OLD,
        statistics['resamples'],statistics['bootstrap_seed'])
    if set(comparison) != set(COMPARISON_KEYS):
        raise ValueError('Frozen matched-comparison output schema differs')
    return computed,items,comparison,counts


def replay(root=ROOT):
    root=Path(root).resolve()
    base=load_module(public_file(root,'tools/replay_results.py'),'openai_public_base_replay')
    previous=load_module(public_file(root,'tools/replay_extension_results.py'),'openai_public_previous_replay')
    manifest=base.verify_public_hashes(root)
    provenance=read(public_file(root,'provenance/'+EXTENSION+'/public_release.json'))
    runtime=require_final_runtime(provenance)
    for version in runtime.values():
        arithmetic_for(version,previous)
    sources=provenance['source_code_sha256']
    if set(sources) != set(SOURCE_FILES):
        raise ValueError('Frozen source whitelist differs')
    for relative,expected in sources.items():
        if sha(public_file(root,relative)) != expected:
            raise ValueError('Frozen API source hash differs')
    # The existing release verifier pins the original metric module as well.
    base.load_analysis(root)
    analysis=load_module(public_file(root,EXTENSION+'/analyze.py'),'openai_frozen_public_analysis')
    comparator=load_module(public_file(root,EXTENSION+'/compare_codex.py'),'openai_frozen_public_comparator')
    protocol=read(public_file(root,'provenance/'+EXTENSION+'/protocol.json'))
    if (protocol['models'] != MODELS or list(protocol['models']) != list(MODELS)
            or protocol['main_pairs'] != 300 or protocol['batch_size'] != 10
            or protocol['counts']['main_prediction_records'] != 10800
            or protocol['counts']['total_planned_calls'] != 1080):
        raise ValueError('API panel or planned dataset cardinality differs')
    prefix='results/'+EXTENSION+'/'
    records=read(public_file(root,prefix+'per_prediction.json'))
    jobs=read(public_file(root,prefix+'jobs.json'))
    reference=previous.reference_contract(read(public_file(root,'results/per_prediction.json')))
    computed,items,comparison,counts=compute_results(records,jobs,protocol,reference,
        read(public_file(root,'results/summary.json')),read(public_file(root,'results/per_item.json')),
        read(public_file(root,'results/jobs.json')),base,previous,analysis,comparator,runtime)
    base.compare(items,read(public_file(root,prefix+'per_item.json')))
    base.compare(computed,read(public_file(root,prefix+'summary.json')))
    base.compare(comparison,read(public_file(root,prefix+'comparison_codex_matched.json')))
    report={'status':'PASS','scope':'Public labels and aggregate arithmetic; not private evidence or serving attestation',
        'recorded_runtime':runtime,'host_python':'.'.join(map(str,sys.version_info[:3])),
        'arithmetic':'Module-local recorded CPython 3.11 or 3.12 numeric summation; exact comparisons, no tolerance or rounding',
        **counts,'verified_public_files':len(manifest['sha256']),'new_model_calls':0,
        'all_released_summary_fields_exactly_reproduced':True,'per_item_records_exactly_reproduced':True,
        'matched_codex_comparisons_exactly_reproduced':True,
        'limits':['Labels cannot independently attest extraction from omitted raw responses.',
            'Cumulative attempt counts include disjoint administrative stages and may exceed two; omitted historical paths prevent public verification of stage selection.',
            'Recorded attempt entries may include reservations without proven HTTP dispatch.',
            'API-reported costs and missingness are arithmetic evidence, not independently verified provider bills.',
            'Own-S dependent revisions and all planned failures remain in the denominators.',
            'Matched aliases do not attest identical model weights or establish a causal interface effect.',
            'Automatic-reference agreement is not human-validated linguistic accuracy.']}
    return computed,comparison,report


def output_path(root,value=None):
    root=Path(root).resolve()
    build=root/'build'
    output=Path(value).absolute() if value is not None else build/'openai_api_replay'
    reject_link_components(root,build)
    reject_link_components(root,output)
    build,output=build.resolve(),output.resolve()
    if (not build.is_relative_to(root) or not output.is_relative_to(root)
            or output==build or not output.is_relative_to(build) or output.exists()):
        raise ValueError('Choose a NEW output subdirectory inside the public root build/')
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT,help='Completed public release root')
    parser.add_argument('--output',type=Path,help='New subdirectory inside the public root build directory')
    args=parser.parse_args()
    root=args.root.resolve()
    output=output_path(root,args.output)
    summary,comparison,report=replay(root)
    output.mkdir(parents=True,exist_ok=False)
    for name,value in (('summary',summary),('comparison_codex_matched',comparison),('verification',report)):
        (output/(name+'.json')).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
