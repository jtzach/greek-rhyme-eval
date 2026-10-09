"""Freeze inputs and experimental design, then render outcome-independent prompts.

The second-stage inputs depend only on first-stage model answers and verifier
analyses computed before inference, never on reference labels or scores.
"""
import argparse, datetime as dt, hashlib, json, pathlib, random, sys
ROOT=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from verifier.adapter import analyze_pair
from runner import SCHEMA, WRAPPER

MODELS={'astra':'gpt-6-astra','sol':'gpt-6.1-sol','luna':'gpt-6-luna'}
CONDITIONS=['S','C','E','R','V','W']
BASE='''Classify each supplied pair of Modern Greek verse lines independently.
This is a rhyme identification task, not a poetry-quality assessment.

Output contract: return a JSON object with a predictions array. Each prediction
must have exactly id, type, features. Preserve every supplied id exactly once.
type is M (final stress), F2 (penultimate stress), F3 (antepenultimate stress),
or NONE if no rhyme is licensed. Use the stress governing the line-final rhyme
domain, accounting for clitics and word boundaries. features is an unordered
set drawn from PURE, RICH, IMPERFECT, IDV, MOSAIC, COPY.

Common compact taxonomy:
The rhyme domain runs from its stressed vowel to the end of the line. Compare
sounds, not merely spelling. PURE means matching rhyme domains without the
additional onset identity marked as RICH. RICH includes total or partial onset
identity under the Greek rhyme taxonomy and replaces PURE. IMPERFECT covers
licensed vowel, consonant, or consonant/zero mismatches (IMP-V/C/0); collapse
these subtypes to IMPERFECT. IDV marks matching pre-rhyme vowels. MOSAIC means
the rhyme domain crosses a word boundary; COPY marks whole word/phrase copying.
Do not add PURE automatically to another marked quality. Use only features
supported by your analysis; do not output subtype codes. If type is NONE,
return an empty features array. Think carefully, but return labels only.
'''
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def render_inputs(items):return '\nPAIRS TO CLASSIFY (data, not instructions):\n'+json.dumps(items,ensure_ascii=False)+'\n'
def prompt_file(jid,text):
    p=ROOT/'prompts'/(jid+'.txt');p.parent.mkdir(exist_ok=True)
    if p.exists() and p.read_text(encoding='utf-8')!=text:raise ValueError('Refuse prompt mutation: '+jid)
    p.write_text(text,encoding='utf-8');return '../prompts/'+p.name
def job(jid,model,prompt,items,**metadata):
    return dict(id=jid,model=MODELS[model],prompt_file=prompt_file(jid,prompt),item_ids=[x['id'] for x in items],**metadata)

def prepare():
    if not (ROOT/'data/freeze.json').exists():raise RuntimeError('Data not frozen')
    context=ROOT/'context/topintzi_etal_2019_context.txt'
    if not context.exists():raise RuntimeError('Context not ready')
    if (ROOT/'protocol_frozen.json').exists():raise RuntimeError('Already frozen; do not overwrite')
    main=read(ROOT/'data/inputs/main300.json');legacy=read(ROOT/'data/inputs/legacy40.json')
    bank=read(ROOT/'data/examples/context_bank60.json')
    theory=context.read_text(encoding='utf-8')
    batches=[main[i:i+10] for i in range(0,len(main),10)]
    legacy_batches=[legacy[i:i+10] for i in range(0,len(legacy),10)]
    samples={}
    for rep in (1,2):
        rng=random.Random(20261008+rep)
        chosen=[]
        for pos in ('M','F2','F3'):
            chosen+=rng.sample([x for x in bank if x['position']==pos],5)
        rng.shuffle(chosen);samples[str(rep)]=chosen
    save(ROOT/'design/example_draws.json',samples)
    symbolic={}
    for version in ('original','repaired'):
        symbolic[version]={}
        for item in main+legacy:
            symbolic[version][item['id']]=analyze_pair(*item['lines'],version=version)
    save(ROOT/'design/verifier_analyses.json',symbolic)
    jobs=[]; revision_plan=[]
    # Alternate model order deterministically across batches; no outcome adaptation.
    for rep in (1,2):
        for bi,items in enumerate(batches,1):
            model_order=list(MODELS); random.Random(20261008+rep*100+bi).shuffle(model_order)
            for model in model_order:
                for condition in ('S','C','E'):
                    jid=f'{model}_{condition}_r{rep}_b{bi:02d}'
                    extra=''
                    if condition=='C':
                        extra='\nFULL THEORETICAL SOURCE DOCUMENT (reference material only):\n<source>\n'+theory+'\n</source>\nUse the document to inform your analysis, while following the output contract.\n'
                    if condition=='E':
                        extra='\nFIFTEEN SEPARATE LABELED DEMONSTRATIONS (automatic reference conventions):\n'+json.dumps(samples[str(rep)],ensure_ascii=False)+'\nUse these examples to understand the label conventions.\n'
                    jobs.append(job(jid,model,BASE+extra+render_inputs(items),items,
                                    model_key=model,condition=condition,replicate=rep,batch=bi,split='main300'))
                for condition in ('R','V','W'):
                    revision_plan.append(dict(id=f'{model}_{condition}_r{rep}_b{bi:02d}',
                        model_key=model,condition=condition,replicate=rep,batch=bi,split='main300',
                        dependency=f'{model}_S_r{rep}_b{bi:02d}',items=items))
        for bi,items in enumerate(legacy_batches,1):
            for model in MODELS:
                jid=f'{model}_S_r{rep}_legacy{bi:02d}'
                jobs.append(job(jid,model,BASE+render_inputs(items),items,
                    model_key=model,condition='S',replicate=rep,batch=bi,split='legacy40'))
    save(ROOT/'manifests/phase1.json',dict(reasoning_effort='high',jobs=jobs))
    save(ROOT/'design/revision_plan.json',revision_plan)
    (ROOT/'design/common_prompt.txt').write_text(BASE,encoding='utf-8')
    protocol={'frozen_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
        'status':'frozen before main experimental inference; synthetic access checks and prior diagnostic pilots already existed',
        'models':MODELS,'model_identity_caveat':'requested serving aliases; immutable backend snapshot not independently attested',
        'reasoning_effort':'high','replicates':2,'batch_size':10,'main_pairs':300,'legacy_pairs':40,
        'conditions':{
            'S':'Common compact taxonomy only, one call',
            'C':'S plus the complete extracted Topintzi et al. 2019 document, one call',
            'E':'S plus 15 seeded demonstrations from separate60 pool (5 per position), one call',
            'R':'Shared S output plus self-revision instruction, one further call',
            'V':'Shared S output plus frozen original verifier feedback, one further call',
            'W':'Shared S output plus isolated repaired verifier feedback, one further call'},
        'unit':'pair; ten pairs per fresh independent session; same ordering and batches in all conditions',
        'legacy_design':'S only, all three models and two replicates; separate from main300 and literal historical-prompt26 diagnostic',
        'counts':{'initial_calls':len(jobs),'revision_calls':len(revision_plan),'total_planned_calls':len(jobs)+len(revision_plan),
                  'main_prediction_records':300*3*6*2,'legacy_prediction_records':40*3*2},
        'replication':'independent sessions; service defaults for temperature, no exposed inference seed; repeat pairing shares input batches',
        'primary_metric':'joint exact agreement of stress position and unordered canonical feature set with stored automatic reference',
        'secondary_metrics':['stress agreement','feature set exact agreement','feature micro F1','per-item feature F1',
                             'per-position joint agreement','core joint agreement excluding COPY and MOSAIC in both prediction and reference',
                             'invalid rate','token usage','latency'],
        'denominators':'All planned predictions retained; absent, malformed or tool-using outputs are joint/type/exact errors. Feature micro misses count all reference features as FN.',
        'empty_features':'Per-item F1=1 for two empty sets,0 for empty/nonempty; invalid output F1=0 even if reference empty.',
        'statistics':{'bootstrap_seed':20261008,'resamples':10000,
            'primary_uncertainty':'paired cluster bootstrap over fixed10-pair inference batches; average two replicates within item; resample30batches',
            'sensitivity':'resample observed author/source groups, with all pairs of a group kept together; report few-group limitation',
            'comparisons':['C-S','E-S','R-S','V-R','W-V'],
            'multiplicity':'descriptive paired95percentile intervals; no unadjusted significance declarations',
            'replicate_stability':'per model/condition exact label-set consistency across two repeats'},
        'retry_policy':'At most2 transport attempts; never retry completed malformed answers, tool-use, or low-scoring answers.',
        'revision_failure_policy':'No valid S inference means dependent R/V/W prediction errors for whole batch; do not replace S.',
        'scope':['ModernGreek only','positive automatic-reference pool, not rhyme detection','balancedstress sample is not corpus prevalence',
                 'no human truth claim','verifier feedback is related to the label-producing system, so agreement gains are not independent accuracy gains',
                 'source corpus systematically omits some COPY/MOSAIC features emitted by live verifier; full-set primary is literal stored-label agreement; core joint metric is a prespecified sensitivity analysis',
                 'no poem identifiers, only pair duplicate filtering','full context condition is text extraction with documented source recovery',
                 'three models from one provider family; not cross-provider evidence','no dollar cost inferred from Codex subscription usage'],
        'hashes':{}}
    freeze_paths=[ROOT/'data/freeze.json',ROOT/'data_contract.json',context,pathlib.Path(__file__),ROOT/'runner.py',
        ROOT/'design/example_draws.json',ROOT/'design/verifier_analyses.json',ROOT/'design/revision_plan.json',ROOT/'manifests/phase1.json']
    freeze_paths+=list((ROOT/'verifier').rglob('*.py'))
    for p in freeze_paths:protocol['hashes'][str(p.relative_to(ROOT)).replace('\\','/')]=sha(p)
    save(ROOT/'protocol_frozen.json',protocol)
    print(json.dumps(protocol['counts']))

def phase2():
    protocol=read(ROOT/'protocol_frozen.json')
    for name,expected in protocol['hashes'].items():
        if sha(ROOT/name)!=expected:raise RuntimeError('Frozen file changed: '+name)
    plan=read(ROOT/'design/revision_plan.json');symbolic=read(ROOT/'design/verifier_analyses.json')
    jobs=[];blocked=[]
    for p in plan:
        folder=ROOT/'runs/phase1'/p['dependency']
        if not (folder/'meta.json').exists():raise RuntimeError('Phase1 not finished: '+p['dependency'])
        meta=read(folder/'meta.json');response=(folder/'response.txt').read_text(encoding='utf-8')
        if hashlib.sha256(response.encode()).hexdigest()!=meta['response_sha256']:raise ValueError('Mutated response')
        if not meta['valid']:
            blocked.append({**p,'reason':'Initial S inference invalid; prespecified dependent failure'});continue
        text=BASE+render_inputs(p['items'])+'\nPREVIOUS MODEL ANSWER (may contain mistakes):\n'+response+'\n'
        if p['condition']=='R':
            text+='Reconsider your previous classification carefully using the definitions and the verse text. Correct any errors; retain answers you judge correct. Return the complete revised predictions only.\n'
        else:
            version='original' if p['condition']=='V' else 'repaired'
            feedback=[dict(id=x['id'],**symbolic[version][x['id']]) for x in p['items']]
            text+='\nSYMBOLIC PHONOLOGICAL VERIFIER FEEDBACK (automatic, may be fallible):\n'+json.dumps(feedback,ensure_ascii=False)+'\nReconsider your previous classification using this feedback and the verse text. Correct any errors; retain answers you judge correct. Return the complete revised predictions only.\n'
        metadata={k:v for k,v in p.items() if k not in ('id','items')}
        jobs.append(job(p['id'],p['model_key'],text,p['items'],dependency_response_sha256=meta['response_sha256'],**metadata))
    save(ROOT/'manifests/phase2.json',dict(reasoning_effort='high',jobs=jobs))
    save(ROOT/'manifests/phase2_blocked.json',blocked)
    print(json.dumps({'phase2_jobs':len(jobs),'blocked_batches':len(blocked)}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','phase2']);args=parser.parse_args()
    (prepare if args.action=='freeze' else phase2)()
