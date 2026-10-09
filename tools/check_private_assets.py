"""Check separately acquired local inputs; no downloads, execution or inference."""
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True,
                        help='Separate private reconstruction workspace with the documented relative layout.')
    args = parser.parse_args()
    workspace = Path(os.path.abspath(args.workspace)).resolve()
    manifest = json.loads((ROOT / 'provenance/private_assets.json').read_text(encoding='utf-8'))
    outcomes = []
    for relative, record in manifest['assets'].items():
        target = (workspace / relative).resolve()
        if not target.is_relative_to(workspace):
            raise ValueError('Input escapes workspace')
        status = 'missing'
        if target.is_file():
            status = 'match' if hashlib.sha256(target.read_bytes()).hexdigest() == record['sha256'] else 'hash_mismatch'
        outcomes.append({'asset': relative, 'status': status})
    complete = all(row['status'] == 'match' for row in outcomes)
    print(json.dumps({'all_listed_assets_match': complete, 'assets': outcomes,
        'scope': 'Required source-input check only. Original frozen generated artifacts, private run evidence and omitted repair implementation are not supplied.'}, indent=2))
    raise SystemExit(0 if complete else 2)

if __name__ == '__main__':
    main()
