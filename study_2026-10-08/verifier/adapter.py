"""One pair-level contract for frozen and repaired verifiers.

Canonical features: PURE, RICH, IMPERFECT, IDV, MOSAIC, COPY. No inferred
PURE feature for RICH. Exact imperfect subtypes remain in raw outputs.
"""
import builtins
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from types import ModuleType

SOURCE = Path(__file__).resolve().parents[2] / 'repository_review_2026-09-29' / 'src'
FEATURES = frozenset({'PURE', 'RICH', 'IMPERFECT', 'IDV', 'MOSAIC', 'COPY'})


def _load_frozen(name, engine=None):
    """Execute unchanged source with module-local imports; no sys.path mutation.

    Resolving greek_phonology locally also covers the function-local imports in
    the frozen feedback module without replacing any global sys.modules entry.
    """
    path = SOURCE / (name + '.py')
    module = ModuleType('_study_frozen_' + name)
    module.__file__ = str(path)
    if engine is not None:
        def local_import(import_name, globals=None, locals=None, fromlist=(), level=0):
            if import_name == 'greek_phonology' and level == 0:
                return engine
            return builtins.__import__(import_name, globals, locals, fromlist, level)
        module.__dict__['__builtins__'] = dict(vars(builtins), __import__=local_import)
    exec(compile(path.read_text(encoding='utf-8'), str(path), 'exec'), module.__dict__)
    return module


@lru_cache(maxsize=2)
def get_engine(version='original'):
    if version == 'original':
        return _load_frozen('greek_phonology')
    if version == 'repaired':
        return import_module('.repaired.greek_phonology', __package__)
    raise ValueError("version must be 'original' or 'repaired'")


@lru_cache(maxsize=2)
def get_feedback(version='original'):
    engine = get_engine(version)
    if version == 'original':
        return _load_frozen('verification_utils', engine)
    return import_module('.repaired.verification_utils', __package__)


def normalize_features(values):
    """Normalize only known study codes; generic IMP-0 remains unspecified."""
    features = set()
    for value in values:
        value = str(value).upper()
        if value in {'IMP', 'IMP-C', 'IMP-V', 'IMP-0', 'IMP-0F', 'IMP-0M'}:
            value = 'IMPERFECT'
        elif value == 'MOS':
            value = 'MOSAIC'
        elif value in {'TR', 'PR', 'TR-S', 'TR-CC', 'PR-C'}:
            value = 'RICH'
        if value in FEATURES:
            features.add(value)
    return sorted(features)


def analyze_pair(line1, line2, version='original'):
    """Analyze two individual lines using the selected verifier, never labels.

    NONE/UNKNOWN are explicit stress_type outcomes for rejected/unreadable
    pairs. For accepted pairs, stress_type is the engine's M/F2/F3 subtype.
    valid records the engine's acceptance; it is not a human validity claim.
    """
    if not isinstance(line1, str) or not isinstance(line2, str):
        raise TypeError('Each line must be a string')
    if '\n' in line1 or '\n' in line2 or '\r' in line1 or '\r' in line2:
        raise ValueError('analyze_pair accepts two individual lines')
    engine = get_engine(version)
    domains = [engine.extract_rhyme_domain(line) for line in (line1, line2)]
    words = [domain['rhyme_domain'].strip('*·.').strip() for domain in domains]
    raw = engine.classify_rhyme_pair(*words)
    quality = raw.get('type', 'UNKNOWN')
    valid = quality not in {'NONE', 'UNKNOWN'}
    feedback = get_feedback(version).find_all_rhymes_in_poem(line1 + '\n' + line2)
    record = feedback[0] if feedback else None
    values = ([quality] + (record or {}).get('features', [])) if valid else []
    stress = raw.get('subtype', 'UNKNOWN') if valid else quality
    if stress not in {'M', 'F2', 'F3', 'NONE', 'UNKNOWN'}:
        stress = 'UNKNOWN'
    return {
        'version': version,
        'stress_type': stress,
        'features': normalize_features(values),
        'valid': valid,
        'details': raw.get('details', raw.get('reason', '')),
        'raw_classification': raw,
        'raw_feedback': record,
        'rhyme_domains': words,
    }
