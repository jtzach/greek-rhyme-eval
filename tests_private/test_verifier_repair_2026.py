"""Engineering invariants: synthetic strings only, no corpus-label oracle."""
import hashlib
import json
from pathlib import Path
import sys
import unicodedata
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'study_2026-10-08'))
from verifier.adapter import analyze_pair, get_engine, get_feedback, normalize_features


class VerifierEngineeringRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = get_engine('original')
        cls.repaired = get_engine('repaired')

    def test_source_and_repair_hashes_match_record(self):
        here = ROOT / 'study_2026-10-08' / 'verifier'
        metadata = json.loads((here / 'repair_provenance.json').read_text(encoding='utf-8'))
        for name, hashes in metadata['hashes'].items():
            with self.subTest(name=name):
                # Frozen provenance was recorded on Windows; preserve its bytes
                # while interpreting the relative path on any supported OS.
                old = ROOT / metadata['frozen_source'].replace('\\', '/') / name
                new = here / 'repaired' / name
                self.assertEqual(hashlib.sha256(old.read_bytes()).hexdigest(), hashes['original_sha256'])
                self.assertEqual(hashlib.sha256(new.read_bytes()).hexdigest(), hashes['repaired_sha256'])

    def test_ascii_chi_feature_alias(self):
        self.assertEqual(self.repaired.g2p_greek('χ'), 'X')
        self.assertNotIn('X', self.original.CONSONANT_FEATURES)
        self.assertEqual(self.repaired.CONSONANT_FEATURES['X'], self.repaired.CONSONANT_FEATURES['x'])
        self.assertEqual(self.repaired.CONSONANT_FEATURES['X'], (3, 2))
        # Both differ from f only in place, so the existing compatibility rule
        # must treat the chi and theta probes consistently after encoding repair.
        for word in ('άχα', 'άθα'):
            with self.subTest(word=word):
                result = self.repaired.classify_rhyme_pair(word, 'άφα')
                self.assertEqual((result['type'], result['imperfect_type']), ('IMPERFECT', 'IMP-C'))
        self.assertEqual(self.original.classify_rhyme_pair('άχα', 'άφα')['type'], 'NONE')

    def test_every_consonant_digraph_shared_before_and_after_stress(self):
        expected = {'μπ': 'b', 'ντ': 'd', 'γκ': 'g', 'γγ': 'Ng', 'τσ': 'ts', 'τζ': 'dz'}
        for letters, sound in expected.items():
            with self.subTest(letters=letters):
                self.assertEqual(self.repaired.g2p_greek(letters + 'άλα'), sound + 'ala')
                self.assertEqual(self.repaired.get_rhyme_part_and_onset(letters + 'άλα'), ('ala', sound))
                self.assertEqual(self.repaired.get_rhyme_part_and_onset('ά' + letters + 'α'), ('a' + sound + 'a', ''))
                self.assertEqual(self.repaired.g2p_greek('ά' + letters + 'α'), 'a' + sound + 'a')

    def test_tokens_preserve_digraph_stress_offsets(self):
        units = self.repaired.tokenize_orthography('μπείνα')
        self.assertEqual([(u['t'], u['start'], u['end']) for u in units],
                         [('μπ', 0, 2), ('εί', 2, 4), ('ν', 4, 5), ('α', 5, 6)])
        self.assertEqual(self.repaired.get_rhyme_part_and_onset('μπείνα'), ('ina', 'b'))
        # Decomposed accent input is normalized by both existing public paths.
        decomposed = unicodedata.normalize('NFD', 'μπείνα')
        self.assertEqual(self.repaired.get_rhyme_part_and_onset(decomposed), ('ina', 'b'))
        self.assertEqual(self.repaired.g2p_greek(decomposed), 'bina')

    def test_g2p_existing_policy_unchanged(self):
        probes = ['μπάλα', 'ντάλα', 'γκάλα', 'γγάλα', 'τσάλα', 'τζάλα',
                  'άμπα', 'άγγα', 'αύτα', 'αύδα', 'ευ', 'αυ α', 'παιδιά',
                  '«ΧΑ!», μπείνα;', 'όνομά της', 'ΐ ΰ ϊϋ', '', 'x-X 123']
        for text in probes:
            for ascii_mode in (False, True):
                with self.subTest(text=text, ascii_mode=ascii_mode):
                    self.assertEqual(self.repaired.g2p_greek(text, ascii_mode),
                                     self.original.g2p_greek(text, ascii_mode))

    def test_unrelated_phonology_preserved(self):
        probes = ['άλα', 'άτα', 'αλά', 'ούλα', 'αύτα', 'αύδα', 'παιδιά', 'όνομά της', '']
        for word in probes:
            with self.subTest(word=word):
                self.assertEqual(self.repaired.get_rhyme_part_and_onset(word), self.original.get_rhyme_part_and_onset(word))
                self.assertEqual(self.repaired.detect_stress_position(word), self.original.detect_stress_position(word))
                self.assertEqual(self.repaired.extract_rhyme_domain(word), self.original.extract_rhyme_domain(word))

    def test_structured_imperfect_subtypes_preserved(self):
        cases = [('άτα', 'άζα', 'IMP-C'), ('άτα', 'έτα', 'IMP-V'), ('άστα', 'άτα', 'IMP-0')]
        for first, second, subtype in cases:
            with self.subTest(subtype=subtype):
                raw = self.repaired.classify_rhyme_pair(first, second)
                self.assertEqual(raw['imperfect_type'], subtype)
                # The detail field is deliberately descriptive and has no code.
                self.assertNotIn(subtype, raw['details'])
                repaired = get_feedback('repaired').find_all_rhymes_in_poem(first + '\n' + second)
                original = get_feedback('original').find_all_rhymes_in_poem(first + '\n' + second)
                self.assertEqual(repaired[0]['features'], [subtype])
                self.assertEqual(original[0]['features'], [])
                for version in ('original', 'repaired'):
                    canonical = analyze_pair(first, second, version)
                    self.assertEqual(canonical['features'], ['IMPERFECT'])
                    self.assertEqual(canonical['raw_classification']['imperfect_type'], subtype)

    def test_feature_vocabulary(self):
        self.assertEqual(normalize_features(['TR', 'TR-S', 'PR-C', 'MOS', 'IMP-C', 'IMP-V', 'IMP-0', 'OTHER']),
                         ['IMPERFECT', 'MOSAIC', 'RICH'])
        for version in ('original', 'repaired'):
            identical = analyze_pair('άλα', 'άλα', version)
            self.assertEqual(identical['features'], ['COPY', 'PURE'])
            rich = analyze_pair('κάλα', 'κάλα', version)
            self.assertEqual(rich['features'], ['COPY', 'RICH'])
            self.assertNotIn('PURE', rich['features'])

    def test_adapter_explicit_nonrhyme_and_unknown(self):
        for version in ('original', 'repaired'):
            for first, second, expected in [('αλά', 'άλα', 'NONE'), ('', '', 'UNKNOWN'), ('β', 'δ', 'UNKNOWN')]:
                with self.subTest(version=version, first=first, second=second):
                    result = analyze_pair(first, second, version)
                    self.assertEqual(result['stress_type'], expected)
                    self.assertFalse(result['valid'])
                    self.assertEqual(result['features'], [])
            for first, expected in [('λα', 'M'), ('λάλα', 'F2'), ('λάλαλα', 'F3')]:
                result = analyze_pair(first, first, version)
                self.assertTrue(result['valid'])
                self.assertEqual(result['stress_type'], expected)

    def test_versions_remain_isolated(self):
        # Loading repaired feedback must never overwrite the original functions.
        before = analyze_pair('άμπα', 'άπα', 'original')
        after_repair = analyze_pair('άμπα', 'άπα', 'repaired')
        after = analyze_pair('άμπα', 'άπα', 'original')
        self.assertEqual(before, after)
        self.assertEqual(before['raw_classification']['imperfect_type'], 'IMP-0')
        self.assertEqual(after_repair['raw_classification']['imperfect_type'], 'IMP-C')
        self.assertEqual(self.original.get_rhyme_part_and_onset('μπάλα'), ('ala', 'mp'))
        self.assertEqual(self.repaired.get_rhyme_part_and_onset('μπάλα'), ('ala', 'b'))


if __name__ == '__main__':
    unittest.main()
