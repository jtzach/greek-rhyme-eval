"""Numerically replay released labels using the byte-identical study scorer.

This checks public records, not the unavailable private prompts, model responses
or event logs. It never imports the phonological verifier or invokes inference.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARY_KEYS = ('reference_status', 'counts', 'statistics', 'metric_notes', 'cells',
                'contrasts', 'baselines', 'cost_total', 'baseline_changes')
FEATURES = {'PURE', 'RICH', 'IMPERFECT', 'IDV', 'MOSAIC', 'COPY'}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_public_hashes(root):
    record = read(root / 'provenance/public_manifest.json')
    for relative, expected in record['sha256'].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file() or digest(path) != expected:
            raise ValueError('Public file missing or changed: ' + relative)
    return record


def load_analysis(root):
    path = root / 'study_2026-10-08/analysis.py'
    source = read(root / 'provenance/public_release.json')
    if digest(path) != source['original_analysis_sha256']:
        raise ValueError('Original scoring code hash differs')
    spec = importlib.util.spec_from_file_location('frozen_study_analysis', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def recompute_rows(rows, analysis, baseline=False):
    output = []
    seen = set()
    for row in rows:
        key = ((row['split'], row['version'], row['id']) if baseline else
               (row['split'], row['model'], row['condition'], row['replicate'], row['id']))
        if key in seen:
            raise ValueError('Duplicate record: ' + str(key))
        seen.add(key)
        if row['reference_type'] not in {'M', 'F2', 'F3'}:
            raise ValueError('Unsupported reference type')
        for name in ('reference_features', 'predicted_features'):
            values = row[name]
            if not isinstance(values, list) or len(values) != len(set(values)) or not set(values) <= FEATURES:
                raise ValueError('Invalid feature set')
        if row['valid'] not in (0, 1):
            raise ValueError('Invalid validity flag')
        if row['valid'] and row['predicted_type'] not in {'M', 'F2', 'F3', 'NONE'}:
            raise ValueError('Invalid prediction type')
        reference = {'position': row['reference_type'], 'features': row['reference_features']}
        prediction = {'type': row['predicted_type'], 'features': row['predicted_features']}
        metrics = analysis.score_prediction(reference, prediction, valid=bool(row['valid']))
        if any(row.get(name) != value for name, value in metrics.items()):
            raise ValueError('Stored metrics disagree with labels: ' + str(key))
        output.append({**row, **metrics})
    return output


def compare(actual, expected, path=''):
    """Exact JSON comparison, with no numeric tolerance or omitted result field."""
    if isinstance(actual, dict) and isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise ValueError('Result keys differ at ' + path)
        for key in actual:
            compare(actual[key], expected[key], path + '/' + key)
    elif isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError('Result lengths differ at ' + path)
        for index, (left, right) in enumerate(zip(actual, expected)):
            compare(left, right, path + '/' + str(index))
    elif type(actual) is not type(expected) or actual != expected:
        raise ValueError('Result value differs at ' + path)


def replay(root):
    manifest = verify_public_hashes(root)
    analysis = load_analysis(root)
    records = recompute_rows(read(root / 'results/per_prediction.json'), analysis)
    baselines = recompute_rows(read(root / 'results/baseline_items.json'), analysis, baseline=True)
    items = analysis.mean_replicates(records, expected_replicates=2)
    compare(items, read(root / 'results/per_item.json'))
    jobs = read(root / 'results/jobs.json')
    protocol = read(root / 'provenance/protocol_frozen.json')
    computed = analysis.build_summary(records, items, jobs, baselines, protocol,
        resamples=protocol['statistics']['resamples'], seed=protocol['statistics']['bootstrap_seed'])
    computed = {key: computed[key] for key in SUMMARY_KEYS}
    compare(computed, read(root / 'results/summary.json'))
    return computed, analysis, {
        'status': 'PASS', 'scope': 'Public label/aggregate numerical consistency only; not private evidence replay',
        'prediction_records': len(records), 'item_condition_records': len(items), 'baseline_records': len(baselines),
        'job_records': len(jobs), 'verified_public_files': len(manifest['sha256']),
        'all_released_summary_fields_exactly_reproduced': True,
        'per_item_records_exactly_reproduced': True, 'new_model_calls': 0,
        'original_analysis_sha256': digest(root / 'study_2026-10-08/analysis.py'),
        'limits': ['Labels were released separately from raw responses; their extraction cannot be independently replayed here.',
                   'Usage/latency aggregate records are checked through arithmetic, not underlying events.',
                   'Hashes detect divergence from this release manifest, not a coordinated alteration of records and manifest.',
                   'Automatic-reference agreement is not human-validated rhyme accuracy.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'build/replay')
    parser.add_argument('--figures', action='store_true', help='Also render four aggregate figures (requires Matplotlib).')
    args = parser.parse_args()
    output = args.output.resolve()
    build = (ROOT / 'build').resolve()
    if not output.is_relative_to(build) or output == build or output.exists():
        raise ValueError('Choose a NEW output subdirectory inside build/, e.g. --output build/replay2')
    summary, analysis, report = replay(ROOT)
    output.mkdir(parents=True, exist_ok=False)
    analysis.write_json(output / 'summary.json', summary)
    analysis.write_json(output / 'verification.json', report)
    analysis.latex_export(summary, output / 'paper_results.tex')
    if args.figures:
        analysis.create_figures(summary, output / 'figures')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
