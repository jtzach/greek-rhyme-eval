# Public records

All files are UTF-8 JSON. None contains verse text, theoretical prose, raw answers, prompts, account/session IDs or local machine paths. The synthetic fixtures in `data/examples/` are independent test examples.

| File | Rows | Meaning |
|---|---:|---|
| `results/per_prediction.json` | 11,040 | One pair × model × condition × repetition; automatic reference and predicted labels, validity and original metrics |
| `results/per_item.json` | 5,520 | Two repetitions averaged within each pair/model/condition; paired stability fields retained |
| `results/baseline_items.json` | 680 | Original/repaired verifier labels for 300 main + 40 legacy pairs |
| `results/jobs.json` | 1,104 | Aggregate usage/latency for each committed 10-pair job, including retained attempts |
| `results/summary.json` | — | All released cell metrics, intervals, contrasts, baselines, stability and cost accounting |

`id`, `job_id`, `batch`, `condition` and `replicate` are experimental design identifiers. They are not provider request/thread identifiers. `group` is an opaque source-group code; the unpublished mapping replaced original source/author strings in lexicographic order so seeded bootstrap draws remain identical. Preserve record ordering and these codes during exact replay.

Canonical reference stress types are M/F2/F3. Valid model prediction types are M/F2/F3/NONE. UNKNOWN is not a valid model type; it can only describe an unknown engine outcome retained as invalid under the baseline contract. Features are PURE, RICH, IMPERFECT, IDV, MOSAIC and COPY. Labels are categorical observations, not human annotations. `legacy_subset` identifies the original golden26/complementary14 allocation; the historical word “golden” does not mean human verification.

Per-prediction response hashes were removed. Private event provenance cannot be recovered from these files. The public summary omits the separate original-prompt Astra26 diagnostic, original generation timestamp and local historical-result path; it retains every field that the public numerical replay claims to reproduce. The original summary's hash remains in `provenance/public_release.json` as a source record.

`provenance/public_manifest.json` records the bytes consumed by replay. `provenance/source_records.json` identifies byte-identical original code and the portable extraction derivative. The historical freeze records concern the original private experiment: they intentionally reference omitted artifacts and do not certify this source-only release as a full evidence package.

## Extension records

`results/extension_2026-10-10/` adds 28,800 prediction rows, 14,400 repeat-averaged rows, 2,880 planned aggregate jobs, the complete extension summary and matched original comparisons. It reuses the existing public opaque group codes. Blocked revisions remain in planned denominators. Source-response hashes and remote IDs are omitted. The new adapter verifies all released main-extension scientific fields exactly. The separately named `synthetic_format_load_summary.json` contains post hoc diagnostic observations outside those scores and denominators; its artificial inputs and exact request specification are under `docs/diagnostics/`.

`posthoc_repair_localization.json` adds descriptive V/W transition counts for all eleven models, conditional on the published 50/250 partition of opaque item IDs by changed/matching `raw_feedback`. It includes only model IDs, item memberships, counts, semantics and relative source hashes; source feedback is omitted. These group labels do not imply unchanged complete verifier objects or prompts, and the public replay cannot independently regenerate membership from omitted feedback.
