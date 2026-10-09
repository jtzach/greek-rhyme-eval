"""Check blinding, exclusion boundaries and exact provenance before data freeze."""
import hashlib
import json
from pathlib import Path
from prepare_dataset import ROOT, STUDY, SOURCE, pair_key, near_pair, canonical_label


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


assert pair_key(['Καλά!', 'Μαλά.']) == pair_key(['μαλα', 'καλα'])
assert near_pair(pair_key(['Ο ήλιος ανατέλλει', 'Στο δάσος ο ήχος μένει']), pair_key(['στο δασος ο ηχος μενει', 'Ο ηλιος ανατελλει.']))
assert not near_pair(pair_key(['Ο ήλιος ανατέλλει', 'Στο δάσος ο ήχος μένει']), pair_key(['Ο ήλιος ανατέλλει', 'Αλλιώτικη ολόκληρη η δεύτερη φράση']))
assert canonical_label('F3-IMP-C-IMPERFECT-IDV') == {'position':'F3','features':['IDV','IMPERFECT']}
assert canonical_label('M-TR-S-IDV') == {'position':'M','features':['IDV','RICH']}
assert canonical_label('F2-MOS-IDV-2W') == {'position':'F2','features':['IDV','MOSAIC']}
assert canonical_label('M-IDV') == {'position':'M','features':['IDV']}
assert canonical_label('M-RICH')['features'] == ['RICH']

refs = {k:read(STUDY/f'data/references/{k}.json') for k in ('main300','examples60','legacy40')}
corpus = read(SOURCE/'data/complete_corpus_enhanced.json')
pilot = read(ROOT/'evaluation/gold/labels.json')
metadata = read(STUDY/'data/selection_metadata.json')
for relative, digest in metadata['source_files_sha256'].items():
    assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest() == digest
for split, expected in [('main300',300),('examples60',60),('legacy40',40)]:
    inputs = read(STUDY/f'data/inputs/{split}.json')
    manifest = read(STUDY/f'data/manifests/{split}.json')
    assert len(inputs)==len(refs[split])==len(manifest)==expected
    assert len({r['id'] for r in inputs})==expected
    for i,(inp,ref,item) in enumerate(zip(inputs,refs[split],manifest)):
        assert set(inp)=={'id','lines','text'}, 'Unexpected potentially label-bearing input metadata'
        assert set(item)=={'id','input_file','input_index'}
        assert inp['id']==ref['id']==item['id'] and item['input_index']==i
        assert inp['lines']==ref['lines'] and inp['text']=='\n'.join(ref['lines'])
        assert ref['reference']==canonical_label(ref['classification'])
        if split!='legacy40':
            original=corpus[ref['provenance']['source_group']]['entries'][ref['provenance']['source_entry_index']]
            assert original['rhyme_pair']==ref['lines']
            assert original['classification']==ref['classification']
            assert original.get('author_code')==ref['provenance'].get('author_code')
    if split!='legacy40':
        for position in ('M','F2','F3'):
            assert sum(r['reference']['position']==position for r in refs[split])==expected//3

selected=refs['main300']+refs['examples60']
selected_keys=[pair_key(r['lines']) for r in selected]
blocked_keys=[pair_key(r['lines']) for r in refs['legacy40']]+[pair_key(r['rhyme_pair']) for r in pilot]
cross_checks=within_checks=0
for i,key in enumerate(selected_keys):
    for prior in blocked_keys:
        assert not near_pair(key,prior), 'New item overlaps historical data'
        cross_checks+=1
    for prior in selected_keys[:i]:
        assert not near_pair(key,prior), 'New splits overlap internally/across test and examples'
        within_checks+=1
bank=read(STUDY/'data/examples/context_bank60.json')
assert len(bank)==60 and {x['id'] for x in bank}=={x['id'] for x in refs['examples60']}
for ex,ref in zip(bank,refs['examples60']):
    assert ex['lines']==ref['lines'] and ex['position']==ref['reference']['position'] and ex['features']==ref['reference']['features']
result={'status':'passed','source_hashes_checked':len(metadata['source_files_sha256']),
        'new_pairs':360,'legacy_pairs':40,'pilot_exclusions':100,'cross_historical_checks':cross_checks,
        'within_new_split_checks':within_checks,'input_whitelist_passed':True,'provenance_roundtrip_passed':True,
        'model_outputs_read':False}
(STUDY/'preparation/validation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result))
