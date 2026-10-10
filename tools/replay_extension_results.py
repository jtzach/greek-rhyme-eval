"""Offline numerical replay for the eight-model label-only extension release.

This adapter never calls inference, opens private inputs, or invokes the frozen
analysis/preparation CLIs. Raw-response extraction and serving are not attested.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
EXTENSION = 'extension_2026-10-10'
CONDITIONS = ('S', 'C', 'E', 'R', 'V', 'W')
SUMMARY_KEYS = ('reference_status', 'counts', 'statistics', 'metric_notes', 'scope',
                'cells', 'contrasts', 'cost_total')
COMPARISON_KEYS = ('interface_caveat', 'cells', 'matched_condition_contrasts')
METRIC_KEYS = ('valid', 'stress', 'feature_exact', 'joint', 'feature_item_f1',
               'core_joint', 'feature_tp', 'feature_fp', 'feature_fn',
               'none_with_nonempty_features', 'predicted_none')
PREDICTION_KEYS = ('id', 'split', 'model', 'model_requested', 'condition', 'replicate',
    'batch', 'group', 'job_id', 'status', 'reference_type', 'reference_features',
    'predicted_type', 'predicted_features') + METRIC_KEYS
JOB_KEYS = ('job_id', 'phase', 'split', 'model', 'model_requested', 'condition',
    'replicate', 'batch', 'items', 'status', 'dependency', 'attempts', 'attempts_with_usage',
    'all_attempt_seconds', 'final_attempt_seconds', 'incomplete_latency', 'incomplete_usage',
    'input_tokens', 'cached_input_tokens', 'cache_write_input_tokens', 'output_tokens',
    'reasoning_output_tokens', 'reported_cost_usd', 'attempts_reporting_cost', 'cost_is_complete')
STATUSES = {'valid', 'invalid_format', 'transport_or_interrupted_failure',
            'tool_use', 'blocked_by_initial_failure'}
ARITHMETIC = {'original':'CPython 3.11 left-to-right numeric sum',
    'extension':'CPython 3.12 compensated float sum with direct integer additions',
    'scope':'Default-zero start, finite built-in numeric values and portable signed-32-bit integer fast paths used by the validated records; module-local bindings only; frozen bytes and exact comparisons unchanged',
    'source':'https://github.com/python/cpython/blob/v3.12.14/Python/bltinmodule.c'}


def sum_312(values, start=0):
    """Extension wave: Neumaier float compensation as in CPython 3.12.

    Integer terms in its float path are added directly. This implements only
    the finite built-in scalar domain used by the frozen analyses.
    """
    total, correction = start, 0.0
    if type(start) is not int or start != 0:
        raise ValueError('Replay sum supports the recorded default-zero start only')
    for value in values:
        if type(value) not in (int,float,bool) or (type(value) is float and not math.isfinite(value)):
            raise ValueError('Replay sums require finite built-in numeric values')
        if type(value) in (int,bool) and not -(2**31) <= value <= 2**31-1:
            raise ValueError('Replay integer term exceeds the portable CPython fast-path range')
        if type(total) is float and type(value) is float:
            combined = total + value
            correction += ((total-combined)+value if abs(total)>=abs(value) else (value-combined)+total)
            total = combined
        else:
            # Int-only prefixes stay exact, and integer terms after the first
            # float follow CPython 3.12's direct-addition path for this dataset.
            if type(total) is int and type(value) in (int,bool) and not -(2**31) <= total+value <= 2**31-1:
                raise ValueError('Replay integer prefix exceeds the portable CPython fast-path range')
            total = total + value
    return total+correction if correction and math.isfinite(correction) else total


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference_contract(original_rows):
    """Use the already released ID, batch, reference and opaque-group contract."""
    result = {}
    fields = ('id', 'batch', 'group', 'reference_type', 'reference_features')
    for row in original_rows:
        if row['split'] != 'main300':
            continue
        value = {key: row[key] for key in fields}
        if row['id'] in result and result[row['id']] != value:
            raise ValueError('Original public reference/group contract is inconsistent')
        result[row['id']] = value
    return result


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
        expected_attempts = (0,) if job['status'] == 'blocked_by_initial_failure' else (1, 2)
        if type(job['attempts']) is not int or job['attempts'] not in expected_attempts:
            raise ValueError('Attempt count differs from the committed failure policy')
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
        for field in JOB_KEYS[11:]:
            value = job[field]
            if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0):
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


def compare_original_public(summary, items, original_summary, original_items, old, resamples, seed):
    """Same paired calculations as the frozen extension; public input locations."""
    old_items = [r for r in original_items if r['split'] == 'main300']
    old_cells = [c for c in original_summary['cells'] if c['split'] == 'main300']
    result = {'interface_caveat': 'Direct OpenRouter API and Codex CLI differ; differences are descriptive and cannot isolate a model-family effect.',
              'cells': [], 'matched_condition_contrasts': []}
    for origin, cells in (('original_codex', old_cells), ('extension_openrouter', summary['cells'])):
        result['cells'] += [{'study': origin, **cell} for cell in cells]
    for model in dict.fromkeys(r['model'] for r in items):
        for old_model in dict.fromkeys(r['model'] for r in old_items):
            for condition in CONDITIONS:
                lhs = [r for r in items if r['model'] == model and r['condition'] == condition]
                rhs = [r for r in old_items if r['model'] == old_model and r['condition'] == condition]
                if len(lhs) != len(rhs):
                    raise ValueError('Original comparison denominators differ')
                for newer, older in zip(sorted(lhs, key=lambda r:r['id']), sorted(rhs, key=lambda r:r['id'])):
                    for key in ('id', 'batch', 'group', 'reference_type', 'reference_features'):
                        if newer[key] != older[key]:
                            raise ValueError('Original comparison reference/batch/group differs')
                result['matched_condition_contrasts'].append({'new_model': model,
                    'original_model': old_model, 'condition': condition, 'contrast': 'new minus original',
                    'joint_batch': old.paired_contrast(lhs, rhs, 'batch', resamples, seed),
                    'core_joint_batch': old.paired_contrast(lhs, rhs, 'batch', resamples, seed, 'core_joint')})
    return result


def replay(root=ROOT):
    root = Path(root).resolve()
    base = load_module(root / 'tools/replay_results.py', 'public_base_replay_extension')
    manifest = base.verify_public_hashes(root)
    provenance = read(root / 'provenance' / EXTENSION / 'public_release.json')
    for relative, expected in provenance['source_code_sha256'].items():
        if sha(root / relative) != expected:
            raise ValueError('Frozen extension source differs: ' + relative)
    old = base.load_analysis(root)
    extension = load_module(root / EXTENSION / 'analyze.py', 'frozen_extension_public_replay')
    old.sum = extension.OLD.sum = extension.sum = sum_312
    protocol = read(root / 'provenance' / EXTENSION / 'protocol.json')
    if len(protocol['models']) != 8 or protocol['main_pairs'] != 300 or protocol['batch_size'] != 10:
        raise ValueError('Public extension panel or dataset cardinality differs')
    result_path = root / 'results' / EXTENSION
    records = base.recompute_rows(read(result_path / 'per_prediction.json'), old)
    jobs = read(result_path / 'jobs.json')
    counts = validate_design(records, jobs, protocol, reference_contract(read(root / 'results/per_prediction.json')))
    items = old.mean_replicates(records, expected_replicates=2)
    base.compare(items, read(result_path / 'per_item.json'))
    statistics = protocol['statistics']
    computed = extension.build_summary(records, items, jobs, protocol,
        resamples=statistics['resamples'], seed=statistics['bootstrap_seed'])
    computed = {key:computed[key] for key in SUMMARY_KEYS}
    base.compare(computed, read(result_path / 'summary.json'))
    comparison = compare_original_public(computed, items, read(root / 'results/summary.json'),
        read(root / 'results/per_item.json'), old, statistics['resamples'], statistics['bootstrap_seed'])
    base.compare(comparison, read(result_path / 'comparison_original.json'))
    report = {'status':'PASS', 'scope':'Public label/aggregate arithmetic only; not private response or serving attestation',
        'arithmetic':ARITHMETIC,'host_python':'.'.join(map(str,sys.version_info[:3])),
        **counts, 'verified_public_files':len(manifest['sha256']), 'new_model_calls':0,
        'all_released_extension_summary_fields_exactly_reproduced':True,
        'per_item_records_exactly_reproduced':True, 'original_comparisons_exactly_reproduced':True,
        'reported_study_cost_usd':computed['cost_total']['reported_cost_usd'],
        'synthetic_preflight_cost_usd':provenance['synthetic_preflight']['reported_cost_usd'],
        'limits':['Omitted responses prevent independent replay of label extraction.',
                  'Released usage/cost/latency aggregates are arithmetic evidence, not provider billing attestation.',
                  'Opaque source groups preserve the released clustering, not poem-disjoint sampling.',
                  'API-versus-Codex comparisons confound interface, serving and reasoning settings.',
                  'Automatic-reference agreement is not human-validated linguistic accuracy.']}
    return computed, comparison, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,default=ROOT/'build/extension_replay')
    args = parser.parse_args()
    output = args.output.resolve()
    build = (ROOT / 'build').resolve()
    if not output.is_relative_to(build) or output == build or output.exists():
        raise ValueError('Choose a NEW output subdirectory inside build/')
    summary, comparison, report = replay(ROOT)
    output.mkdir(parents=True, exist_ok=False)
    for name, value in (('summary',summary),('comparison_original',comparison),('verification',report)):
        (output / (name + '.json')).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
