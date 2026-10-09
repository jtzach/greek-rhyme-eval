# Greek Rhyme Eval

Code and public numerical replay for a Greek rhyme study comparing compact instructions, theoretical context, examples, self-revision and phonological feedback.

**The scores measure agreement with automatic reference labels, not human-validated linguistic accuracy.** The main evaluation contains 300 Greek verse pairs, three requested model aliases, six conditions and two repetitions. An additional 40 historical pairs are evaluated separately.

## Main results

Joint agreement requires both the stress class and the complete unordered feature set to match. Percentages below use all 600 predictions per model/condition.

| Condition | Astra | Sol | Luna |
|---|---:|---:|---:|
| S · compact taxonomy | 63.50 | 64.17 | 56.67 |
| C · S + recovered theory text | 51.67 | 51.83 | 56.83 |
| E · S + 15 examples | 67.50 | 67.83 | 53.33 |
| R · self-revision of S | 64.17 | 63.67 | 59.83 |
| V · original verifier feedback on S | 67.50 | 67.83 | 80.00 |
| W · repaired verifier feedback on S | 67.50 | 67.67 | 80.50 |

![Joint agreement with automatic references](assets/main_joint_comparison.png)

Requested aliases were `gpt-6-astra`, `gpt-6.1-sol` and `gpt-6-luna`, with high reasoning effort through Codex CLI. These are serving aliases; immutable backend weights were not independently attested. The recorded inference dates were 8–9 October 2026.

The full theory condition reduced Astra/Sol agreement, whereas verifier feedback improved Luna most. These observations do not establish that the theory is linguistically wrong or that feedback improves human-judged accuracy: the reference labels and verifier share lineage, and their feature conventions differ in known ways. See [design and interpretation](docs/DESIGN.md).

![Paired differences with descriptive bootstrap intervals](assets/paired_gains_intervals.png)

## Reproduce the public numbers

Python 3.11+ is required. Numerical replay uses NumPy 2.4.1; optional plots use Matplotlib 3.10.8. The original numerical environment was Python 3.11.9 on Windows. Public CI runs synthetic tests and numerical replay without model credentials.

```bash
python -m pip install -e .
python -B -m unittest discover -s tests -v
python -B tools/replay_results.py
```

The last command recomputes item metrics, averages repetitions, runs the fixed 10,000-resample bootstrap and compares **every released summary field exactly**. It writes a verification report and LaTeX table under `build/replay/`. For another run, choose a new output directory, e.g. `--output build/replay2`.

To render the aggregate figures as well:

```bash
python -m pip install -e ".[figures]"
python -B tools/replay_results.py --figures --output build/replay_with_figures
```

Public replay verifies arithmetic from the released records. It cannot independently reconstruct labels from the omitted model responses, attest the provider, or replay private event logs. The frozen full-evidence analysis remains in [`study_2026-10-08/analysis.py`](study_2026-10-08/analysis.py); the public adapter calls its original scoring/statistical functions without bypassing its private-evidence CLI or pretending those inputs are present.

## What is included

| Location | Contents |
|---|---|
| [`results/`](results/) | 11,040 label-only predictions, 5,520 repeat-averaged item records, 680 baseline records, 1,104 aggregate job records and the complete numerical summary |
| [`study_2026-10-08/`](study_2026-10-08/) | Original runner, scorer, dataset/experiment preparation glue and verifier interface; portable theory-text extraction procedure |
| [`src/rhyme_bench/`](src/rhyme_bench/) | Earlier general evaluation library with explicit reference-status checks; separate from the completed study's scorer |
| [`tests/`](tests/) | Synthetic library, runner, scorer and public-adapter tests; no model calls |
| [`provenance/`](provenance/) | Historical hashes, pinned source identity, public sanitization record and release integrity manifest |
| [`docs/RECONSTRUCTION.md`](docs/RECONSTRUCTION.md) | Requirements and boundaries for separate private reconstruction or new inference |

The release excludes poetry text, the upstream unlicensed verifier/corpus, repaired derivative modules, the full theory text/PDF, raw prompts/responses/events, account/session metadata and manuscript PDFs. Source/author group names are replaced by opaque codes; grouping and seeded bootstrap order are preserved. See [data schema](docs/DATA.md) and [rights](RIGHTS.md).

## Attribution

This study builds on Stergios Chatzikyriakidis and Anastasia Natsina's [*LLMs Got Rhyme?* (2026)](https://aclanthology.org/2026.latechclfl-1.9/) and the [Greek_Rhyme repository](https://github.com/StergiosCha/Greek_Rhyme), pinned to `8ee1773d51133213a87b800ef049ad6e2800f7cd`. The C condition used the complete recovered text of Nina Topintzi, Konstantinos Avdelidis and Theodora Valkanou's [*Quantifying Greek Rhyme* (2019)](https://ntopintzi.wixsite.com/webpage/copy-of-research), pages 429–447.

This repository releases research code/results. It does not imply author approval of a manuscript, conference submission or acceptance. Original code licensing is scoped in [RIGHTS.md](RIGHTS.md); no rights to omitted third-party material are granted.
