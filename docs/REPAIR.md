# Engineering repair: procedure and provenance

The W condition used an isolated engineering repair of the pinned upstream verifier. The original/repaired file hashes and recorded scope are in [`provenance/repair_provenance.json`](../provenance/repair_provenance.json).

The repair had three components:

1. Add the uppercase ASCII chi symbol to the consonant-feature lookup using the existing lowercase chi feature values.
2. Use one orthographic tokenizer, retaining source offsets, for display grapheme-to-phoneme conversion and rhyme extraction. Apply the existing consonant-digraph mapping consistently before/after the stressed vowel and convert every symbol in multi-symbol sounds to the established ASCII representation.
3. Preserve the classifier's structured imperfect-rhyme subtype in feedback, rather than recovering a code from descriptive prose. Keep an unspecified zero-mismatch subtype unspecified.

Existing stress, synizesis, consonant compatibility, mosaic and IDV policies were retained. This is an interface/representation consistency repair, not a phonologically validated replacement. Synthetic invariant tests are included in `tests_private/test_verifier_repair_2026.py`; they need separately acquired original/repaired modules and a matching provenance file. They do not use corpus labels as an oracle.

The public package excludes the repaired derivative modules, patch and builder containing source-matching/replacement fragments. Obtain an authorized exact implementation from the maintainers for byte-identical reconstruction. The hashes are identifiers, not permission to redistribute those modules.
