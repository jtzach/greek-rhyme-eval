"""Freeze a label-separated automatic-reference study; no model outputs are read."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
import hashlib
import json
import random
import re
import unicodedata

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / 'study_2026-10-08'
SOURCE = ROOT / 'repository_review_2026-09-29'
COMMIT = '8ee1773d51133213a87b800ef049ad6e2800f7cd'
SEED = 20261008
THRESHOLD = 0.92
BROAD = {'PURE','RICH','IMPERFECT','IDV','MOSAIC','COPY'}
MAP = {'PURE':'PURE','RICH':'RICH','TR':'RICH','PR':'RICH','IMP':'IMPERFECT',
       'IMPERFECT':'IMPERFECT','IDV':'IDV','MOS':'MOSAIC','MOSAIC':'MOSAIC','COPY':'COPY'}
SUBTYPES = {'S','CC','C1','C2','CC1','CC2','V','C','0','0F','0M','2W'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def normalize(text):
    lowered = unicodedata.normalize('NFD', text.casefold())
    unaccented = ''.join(c for c in lowered if unicodedata.category(c) != 'Mn')
    return ' '.join(re.sub(r'[^\w\s]', ' ', unaccented).split())


def pair_key(lines):
    return tuple(sorted(normalize(line) for line in lines))


def similar_line(a, b):
    if a == b:
        return True
    if not a or not b or 2*min(len(a),len(b))/(len(a)+len(b)) < THRESHOLD:
        return False
    return SequenceMatcher(None, a, b, autojunk=False).ratio() >= THRESHOLD


def near_pair(left, right):
    # Accept either line alignment; source ordering is preserved in actual input.
    return ((similar_line(left[0],right[0]) and similar_line(left[1],right[1])) or
            (similar_line(left[0],right[1]) and similar_line(left[1],right[0])))


def canonical_label(classification):
    parts = classification.upper().split('-')
    if parts[0] not in {'M','F2','F3'}:
        raise ValueError('Unsupported reference position')
    features = set()
    for token in parts[1:]:
        if token in MAP:
            features.add(MAP[token])
        elif token not in SUBTYPES:
            raise ValueError('Unsupported reference feature token: '+token)
    if 'PURE' in features and 'RICH' in features:
        raise ValueError('Contradictory PURE/RICH reference')
    return {'position':parts[0], 'features':sorted(features)}


def subtype_codes(classification):
    return re.findall(r'(?:IMP-(?:0F|0M|0|V|C)|(?:TR|PR)-(?:CC1|CC2|C1|C2|CC|S)|IDV-2W)', classification)


def legacy_lines(text):
    if text.count(' / ') != 1:
        raise ValueError('Legacy boundary is not a unique literal space-slash-space')
    return text.split(' / ')


def source_grouping(group, meta, entry):
    author_code = entry.get('author_code')
    # A code is retained literally, never translated into an invented author.
    if author_code is not None and str(author_code).strip():
        return {'source_group':group, 'source_poet_field':meta.get('poet'),
                'author_code':author_code, 'grouping_key':group+'|author_code:'+str(author_code),
                'grouping_basis':'source-scoped literal author_code; code-to-name mapping not inferred'}
    poet = meta.get('poet')
    return {'source_group':group, 'source_poet_field':poet, 'author_code':author_code,
            'grouping_key':group+'|source_poet:'+str(poet) if poet and poet != 'Anemoskala' else None,
            'grouping_basis':'top-level source poet field' if poet and poet != 'Anemoskala' else 'missing'}


def dataset_summary(rows):
    return {'n':len(rows), 'position_counts':dict(Counter(r['reference']['position'] for r in rows)),
            'feature_counts':dict(Counter(f for r in rows for f in r['reference']['features'])),
            'source_group_counts':dict(Counter(r['provenance'].get('source_group') for r in rows)),
            'grouping_key_counts':dict(Counter(str(r['provenance'].get('grouping_key')) for r in rows)),
            'missing_author_code':sum(r['provenance'].get('author_code') in (None,'') for r in rows),
            'missing_grouping_key':sum(r['provenance'].get('grouping_key') is None for r in rows)}


def build():
    if (STUDY/'data/freeze.json').exists():
        raise SystemExit('Refusing to replace frozen data.')
    corpus_path = SOURCE/'data/complete_corpus_enhanced.json'
    corpus = json.loads(corpus_path.read_text(encoding='utf-8'))
    source_paths = [corpus_path, ROOT/'evaluation/gold/labels.json']
    legacy = []
    for subset in ('golden','complementary'):
        path = SOURCE/f'reproduction/identification/{subset}_test_set.json'
        source_paths.append(path)
        for index, row in enumerate(json.loads(path.read_text(encoding='utf-8'))):
            lines = legacy_lines(row['text'])
            legacy.append({'lines':lines, 'original_text':row['text'], 'classification':row['pattern'],
                           'literal_features':row['features'], 'reference':canonical_label(row['pattern']),
                           'subtype_codes':subtype_codes(row['pattern']),
                           'provenance':{'legacy_subset':subset,'legacy_row_index':index,
                                         'source_group':row.get('poet'), 'source_poet_field':row.get('poet'),
                                         'author_code':None,'grouping_key':None,
                                         'grouping_basis':'legacy poet field preserved; no row-level author mapping asserted'}})
    pilot = json.loads(source_paths[1].read_text(encoding='utf-8'))
    blocked = [pair_key(x['lines']) for x in legacy] + [pair_key(x['rhyme_pair']) for x in pilot]
    buckets = defaultdict(list)
    exclusions = Counter()
    for group, meta in corpus.items():
        for index, entry in enumerate(meta['entries']):
            lines = entry.get('rhyme_pair')
            if not isinstance(lines,list) or len(lines)!=2 or not all(isinstance(s,str) and normalize(s) for s in lines):
                exclusions['malformed_or_empty'] += 1
                continue
            classification = entry.get('classification','')
            if 'UNKNOWN' in classification.split('-'):
                exclusions['UNKNOWN'] += 1
                continue
            reference = canonical_label(classification)
            record = {'lines':lines, 'classification':classification,
                      'literal_features':classification.split('-')[1:],
                      'source_features_field':entry.get('features'), 'reference':reference,
                      'subtype_codes':subtype_codes(classification),
                      'provenance':{**source_grouping(group,meta,entry),'source_entry_index':index,
                                    'source_line_indices':entry.get('line_indices'),
                                    'rhyme_positions':entry.get('rhyme_positions'),
                                    'distance':entry.get('distance'),
                                    'context':entry.get('context'), 'phonetic':entry.get('phonetic')},
                      '_key':pair_key(lines)}
            buckets[record['_key']].append(record)
    unique = []
    for bucket in buckets.values():
        if len({x['classification'] for x in bucket}) != 1:
            exclusions['conflicting_normalized_key_records'] += len(bucket)
            continue
        exclusions['normalized_duplicate_records'] += len(bucket)-1
        unique.append(bucket[0])
    # Recover legacy source attribution only through an unambiguous matching
    # released pair; otherwise leave grouping missing rather than guess a poet.
    for row in legacy:
        matches = buckets.get(pair_key(row['lines']), [])
        groups = {r['provenance'].get('grouping_key') for r in matches}
        if matches and len(groups)==1 and None not in groups:
            first = matches[0]['provenance']
            row['provenance'].update({k:first[k] for k in ('source_group','source_poet_field','author_code','grouping_key')})
            row['provenance']['grouping_basis'] = 'unambiguous normalized source-pair match; literal author/source metadata'
            row['provenance']['matching_source_rows'] = [{'source_group':r['provenance']['source_group'],'source_entry_index':r['provenance']['source_entry_index']} for r in matches]
        else:
            row['provenance']['source_match_status'] = 'absent' if not matches else 'ambiguous_group'
    rng = random.Random(SEED)
    selected = {'main300':[], 'examples60':[]}
    accepted_keys = []
    sampling_rejections = Counter()
    for position in ('M','F2','F3'):
        candidates = [x for x in unique if x['reference']['position']==position]
        rng.shuffle(candidates)
        accepted = []
        for candidate in candidates:
            key = candidate['_key']
            if any(near_pair(key,prior) for prior in blocked):
                sampling_rejections[position+'_legacy_or_pilot_overlap'] += 1
                continue
            if any(near_pair(key,prior) for prior in accepted_keys):
                sampling_rejections[position+'_selected_overlap'] += 1
                continue
            accepted.append(candidate)
            accepted_keys.append(key)
            if len(accepted)==120:
                break
        if len(accepted)!=120:
            raise ValueError(f'Insufficient distinct {position} candidates: {len(accepted)}')
        selected['main300'].extend(accepted[:100])
        selected['examples60'].extend(accepted[100:])
    rng.shuffle(selected['main300'])
    rng.shuffle(selected['examples60'])
    selected['legacy40'] = legacy
    for split, rows in selected.items():
        for index, row in enumerate(rows,1):
            row['id'] = {'main300':'pair','examples60':'example','legacy40':'legacy'}[split]+f'_{index:03d}'
            row.pop('_key',None)
        inputs = [{'id':row['id'],'lines':row['lines'],'text':'\n'.join(row['lines'])} for row in rows]
        write_json(STUDY/f'data/inputs/{split}.json',inputs)
        write_json(STUDY/f'data/references/{split}.json',rows)
        write_json(STUDY/f'data/manifests/{split}.json',
                   [{'id':r['id'],'input_file':f'../inputs/{split}.json','input_index':i} for i,r in enumerate(rows)])
    # Explicit example bank for the authorized CAG condition only; never mix this
    # file with held-out labels when assembling model requests.
    example_bank = [{'id':r['id'],'lines':r['lines'],'text':'\n'.join(r['lines']),
                     'position':r['reference']['position'],'features':r['reference']['features']}
                    for r in selected['examples60']]
    write_json(STUDY/'data/examples/context_bank60.json',example_bank)
    metadata = {'seed':SEED,'source_commit':COMMIT,'source_git_blob':'f7fa252c5abff5d5acf6a45e28d346b06ddd7f4f',
                'source_files_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in source_paths},
                'source_total_rows':sum(len(v['entries']) for v in corpus.values()),
                'normalization_exclusions':dict(exclusions),'eligible_normalized_unique_before_overlap_screen':len(unique),
                'candidate_order_rejections_until_quotas_reached':dict(sampling_rejections),
                'blocked_legacy_n':len(legacy),'blocked_pilot_n':len(pilot),
                'splits':{k:dataset_summary(v) for k,v in selected.items()},
                'text_policy':'Original source line strings preserved; input joins the pair with newline. Original legacy space-slash-space text also retained in references.',
                'reference_policy':'Automatic corpus/legacy references, not human ground truth. No model output inspected or used for selection.'}
    write_json(STUDY/'data/selection_metadata.json',metadata)
    contract = {'status':'approved_by_root_awaiting_validation_and_freeze','seed':SEED,'source_commit':COMMIT,
                'splits':{'main300':{'n':300,'per_position':100,'role':'primary held-out evaluation'},
                          'examples60':{'n':60,'per_position':20,'role':'separate labeled context example pool'},
                          'legacy40':{'n':40,'subsets':{'golden':26,'complementary':14},'role':'separate historical benchmark; excluded from new sampling'}},
                'canonical_positions':['M','F2','F3'],'canonical_features':sorted(BROAD),'feature_token_mapping':MAP,
                'subtype_tokens_not_primary_features':sorted(SUBTYPES),
                'canonicalization':'Map recognized literal tokens to broad features, deduplicate/sort, retain source convention. Do not infer PURE. PURE/RICH co-occurrence rejected. Literal classification, literal feature pieces and subtype codes retained.',
                'deduplication':{'normalization':'casefold -> Unicode NFD -> remove combining marks -> replace nonword/nonwhitespace with spaces -> collapse whitespace; sort two normalized lines for unordered key',
                                'near_match':'Either alignment has both normalized line SequenceMatcher(autojunk=False).ratio >=0.92; exact normalized lines match directly',
                                'threshold':THRESHOLD,'blocked_sets':['legacy40','earlier100pilot'],
                                'within_and_between_new_splits':True,
                                'selection':'Normalize/deduplicate corpus in source order; exclude conflicting literal-label keys. For M,F2,F3, shuffle remaining class candidates with one Random(seed), greedily accept120 passing exclusions, first100 main/next20 examples; shuffle main then examples with same RNG.',
                                'scope_limit':'Text-pair near duplication only; different verses with the same rhyme endings may remain. No reliable poem identifiers, so poem-level disjointness cannot be established.'},
                'model_input_schema':{'id':'opaque sequential ID after shuffle','lines':'two original strings','text':'lines joined by newline'},
                'manifest_schema':{'id':'input ID','input_file':'relative path to split input JSON array','input_index':'zero-based index'},
                'reference_schema':{'reference':'{position,features}','classification':'literal source pattern','literal_features':'literal hyphen pieces (legacy supplied list preserved)','subtype_codes':'retained IMP/TR/PR/IDV subtype strings','provenance':'source group, original zero-based row index and available metadata; no invented poem IDs'},
                'paths':{'main_inputs':'data/inputs/main300.json','main_references':'data/references/main300.json','legacy_inputs':'data/inputs/legacy40.json','legacy_references':'data/references/legacy40.json','example_bank':'data/examples/context_bank60.json','metadata':'data/selection_metadata.json'},
                'limitations':['Reference labels are automatic and may be wrong.','No validated negative class.','Author codes retained literally; no author-name decoding inferred.','Some source grouping uses top-level poet field; no poem-level grouping available.','Legacy grouping recovered only through unambiguous normalized-pair match to source metadata; otherwise missing.','Legacy text line breaks are restored only from unique literal space-slash-space separator.']}
    write_json(STUDY/'data_contract.json',contract)
    print(json.dumps(metadata,ensure_ascii=False,indent=2))


if __name__=='__main__':
    build()
