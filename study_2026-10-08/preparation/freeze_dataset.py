"""Seal approved and validated data; no model output or inference is accessed."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
destination = STUDY/'data/freeze.json'
if destination.exists():
    raise SystemExit('Data are already frozen; refusing replacement.')
validation = json.loads((STUDY/'preparation/validation.json').read_text(encoding='utf-8'))
if validation.get('status') != 'passed' or validation.get('new_pairs') != 360:
    raise SystemExit('Complete dataset validation must pass first.')
path = STUDY/'data_contract.json'
contract = json.loads(path.read_text(encoding='utf-8'))
if contract['status'] != 'approved_by_root_awaiting_validation_and_freeze':
    raise SystemExit('Expected the agreed contract.')
contract['status'] = 'frozen'
path.write_text(json.dumps(contract, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
files = [p for p in (STUDY/'data').rglob('*') if p.is_file() and p != destination]
files += [path, STUDY/'preparation/prepare_dataset.py', STUDY/'preparation/validate_dataset.py',
          STUDY/'preparation/freeze_dataset.py', STUDY/'preparation/validation.json']
frozen = {'frozen_utc':datetime.now(timezone.utc).isoformat(), 'approved_seed':20261008,
          'source_commit':contract['source_commit'],
          'scope':'Dataset selection, references, example bank, label-free manifests, contract and preparation code; independent of inference/scoring freeze.',
          'selection_without_model_output_access':True,
          'files_sha256':{p.relative_to(STUDY).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}}
destination.write_text(json.dumps(frozen, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
print(json.dumps({'frozen_utc':frozen['frozen_utc'],'files_frozen':len(files),'main_n':300,'examples_n':60,'legacy_n':40}))
