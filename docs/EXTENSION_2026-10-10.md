# Eight-model direct-API extension

The completed extension was designed and frozen after the original results existed. It preserves the original 300 pairs, six conditions, two repetitions, ten-pair batches, examples, and automatic-reference scoring. It adds 28,800 prediction records and 2,880 planned jobs, with no new Legacy40 evaluation. Invalid outputs and blocked S-dependent revisions remain in all 600-prediction model/condition denominators.

| Exact requested model ID | Pinned provider | Requested reasoning |
|---|---|---|
| anthropic/claude-sonnet-5.5 | anthropic | {"effort": "high"} |
| anthropic/claude-haiku-5.5 | anthropic | {"effort": "high"} |
| google/gemini-3.8-flash | google-ai-studio | {"effort": "high"} |
| deepseek/deepseek-v4.1-flash | deepinfra/fp8 | {"effort": "high"} |
| qwen/qwen3.8-flash | alibaba | {"max_tokens": 4096} |
| mistralai/mistral-large-4-0 | mistral | {"effort": "high"} |
| meta-llama/llama-4-maverick | digitalocean | null |
| z-ai/glm-5.3-flash | fireworks | {"effort": "high"} |

Every main-extension request sets max_tokens=8192 and strict JSON schema. High reasoning is requested where supported; Qwen receives a 4096-token reasoning allocation, and Llama's unsupported reasoning control is omitted. DeepSeek uses DeepInfra FP8. Provider fallbacks, tools, response healing and prompt transforms are disabled. Temperature and seed use provider defaults. These settings do not equate internal computation across models. Exact IDs do not attest immutable weights.

Initial S/C/E inference used 12 workers. Following a local reservation-journal write failure, the process drained submitted work and exited; the unchanged pipeline then resumed with 48 workers to finalize conservative unknown recoveries and run R/V/W revisions. The phase-one configuration retains its original workers=12 provenance; phase two records workers=48. Concurrency changed without changing frozen prompts, model settings or inference failure rules. Recorded recoveries do not prove an HTTP call or observed model-format failure.

The original models used Codex CLI; this extension uses OpenRouter direct HTTP. Differences confound model, interface, provider serving, reasoning controls and quantization where applicable. Scores are agreement with stored automatic references, not independent human-validated accuracy. The positive balanced sample does not establish general rhyme detection or poem-disjoint generalization.

## Offline numerical replay

Use **Python 3.11** for the complete command sequence below, matching the unchanged original-wave replay. Python 3.12 changed built-in float summation. The extension adapter binds the recorded Python 3.12 sum arithmetic locally to its imported analysis modules, so its frozen extension summary is reproduced exactly on tested Python 3.11 and 3.12 hosts. Integer sums and input order are preserved; no result rounding, comparison tolerance or frozen source changes are introduced. The unchanged original CLI remains pinned to Python 3.11. [CPython 3.12 implementation](https://github.com/python/cpython/blob/v3.12.14/Python/bltinmodule.c).

```bash
python -m pip install -e .
python -B -m unittest discover -s tests -v
python -B tools/replay_results.py
python -B tools/replay_extension_results.py
```

The extension adapter uses byte-identical frozen scoring functions and the already released opaque source-group memberships. It verifies 28,800 labels, 14,400 repeat-averaged items, 2,880 planned jobs, all summary fields and every released original-versus-extension paired contrast exactly. Invalid and blocked records cannot be omitted. It creates no model calls and requires no credentials. Results are written to a new build/extension_replay directory.

The frozen runner, preparation and analysis sources are included for reconstruction transparency. Their original CLIs require separately acquired private inputs and raw evidence; public replay does not invoke those CLIs or reconstruct omitted labels from model responses. Source hashes and the derivative protocol are recorded under provenance/extension_2026-10-10/.

## Released observations and cost

results/extension_2026-10-10/ contains categorical labels, repeat-averaged records, aggregate jobs, the complete scientific summary and matched original comparisons. Source/author strings, response hashes, remote generation IDs, precise workflow timestamps and machine paths are omitted. Aggregate recorded-attempt model/provider/finish-reason fields are in provenance/extension_2026-10-10/serving_summary.json; recovered reservations retain null serving fields and do not establish an HTTP call.

Reported cost comes from the API usage.cost field, with missingness retained; it is not independently verified billing. Study reported cost is 23.8481363093 USD; 22 recorded attempt costs remain unreconciled. The separate 16 synthetic access/format checks report 0.069690645 USD and are excluded from study denominators and the study ledger. They are not accuracy evidence. Recovered unknown outcomes remain failures with missing usage/cost and retained reservations; no zero cost or successful dispatch is inferred. Frozen-summary language about physical totals means recorded attempt entries counted once, including such recoveries, not independently verified HTTP-call counts. Public arithmetic cannot independently attest provider billing or raw-response extraction.

Source poetry, theory documents, study prompts/responses/events, account information and credentials are excluded. The separate diagnostic below releases its newly constructed artificial inputs and request specification, without raw HTTP. The existing RIGHTS.md applies; publication of categorical observations grants no rights to omitted source material. This distribution does not imply author approval, conference submission, human gold labels, or anonymous unlinkability.

## Separate post hoc synthetic format/load diagnostic

Sonnet and Mistral were selected after main-study format/length failures were observed. The original 16 preflight checks copied supplied format rows; that does not test classification workload. This separately frozen 24-job probe uses one newly constructed artificial Greek pair, repeated under unique IDs, with no study pairs or linguistic gold. Copy targets are arbitrary serialization targets. Two repetitions compare copy/classify and one/ten items at 8192 output tokens, plus the same classification prompts at 32768 tokens. Only the cap differs between matched classification arms. These easy inputs do not reproduce main-task difficulty or isolate hidden reasoning. The diagnostic does not alter main scores, planned denominators or failed outcomes.

[Exact artificial inputs, request specification and stop rules](diagnostics/synthetic_format_load_specification.json) and [all sanitized job/attempt observations and descriptive cells](../results/extension_2026-10-10/synthetic_format_load_summary.json) are released separately. The specification supports reconstruction; it does not guarantee identical future serving. Raw HTTP, remote generation IDs and the private diagnostic ledger are excluded. Public main-study replay does not claim to reconstruct the omitted diagnostic responses.

**24 recorded attempt entries; 20 HTTP-evidenced response records across 20 jobs.** The frozen `attempted_jobs` field counts jobs with recorded entries, including conservative reservation recoveries; it is not an independently verified HTTP-call count. An interrupted record without HTTP response evidence does not demonstrate model-format behavior.

| Model | Task | Items | Output cap | Valid / 2 | Exact IDs / 2 | Visible text / 2 | Length finishes / 2 | Transport / unknown / 2 | Not dispatched / 2 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| anthropic/claude-sonnet-5.5 | copy | 1 | 8192 | 1 | 1 | 1 | 0 | 1 | 0 |
| mistralai/mistral-large-4-0 | copy | 1 | 8192 | 1 | 1 | 1 | 0 | 1 | 0 |
| anthropic/claude-sonnet-5.5 | copy | 10 | 8192 | 2 | 2 | 2 | 0 | 0 | 0 |
| mistralai/mistral-large-4-0 | copy | 10 | 8192 | 2 | 2 | 2 | 0 | 0 | 0 |
| anthropic/claude-sonnet-5.5 | classify | 1 | 8192 | 2 | 2 | 2 | 0 | 0 | 0 |
| mistralai/mistral-large-4-0 | classify | 1 | 8192 | 1 | 1 | 1 | 1 | 0 | 0 |
| anthropic/claude-sonnet-5.5 | classify | 10 | 8192 | 1 | 1 | 1 | 0 | 1 | 0 |
| mistralai/mistral-large-4-0 | classify | 10 | 8192 | 1 | 1 | 1 | 0 | 1 | 0 |
| anthropic/claude-sonnet-5.5 | classify | 1 | 32768 | 2 | 2 | 2 | 0 | 0 | 0 |
| mistralai/mistral-large-4-0 | classify | 1 | 32768 | 2 | 2 | 2 | 0 | 0 | 0 |
| anthropic/claude-sonnet-5.5 | classify | 10 | 32768 | 2 | 2 | 2 | 0 | 0 | 0 |
| mistralai/mistral-large-4-0 | classify | 10 | 32768 | 2 | 2 | 2 | 0 | 0 | 0 |

All 24 planned terminal outcomes remain reported, including failures and budget stops: 24 jobs have recorded attempts, 0 carry the budget-stop/not-dispatched status, and 0 had a retry stopped by the fixed budget. There are 24 recorded attempts. Unobserved rows retain null observations; they are not API calls or failed linguistic classifications. Counts in the table overlap and measure completion/format only. Diagnostic API-reported cost is 0.19313629 USD; 4 attempt costs remain missing. Its separate fixed ceiling is USD 3, with 0.254052480 USD retained in unknown-outcome reservations. Main study, original 16-check preflight, and diagnostic costs remain separate in [cost-scope provenance](../provenance/extension_2026-10-10/public_release.json). No combined cost total is reported because at least one scope has missing cost coverage. Reported API costs are not independently verified bills; reservations do not establish HTTP dispatch or charges.

The final diagnostic summary records 4 transport/interrupted-unknown job outcomes, with 4 jobs flagged as interrupted unknown outcomes, separately from 1 completed-response format, length, or provider-rejection failures. Missing observations and local journal interruptions are not observed model-format failures. Conservative recovery can create an attempt record from an existing reservation without evidence that HTTP was dispatched. The frozen summary retains these outcomes and missing costs; it does not turn them into successful or linguistically classified responses.

## Post hoc repair localization

This descriptive analysis was selected after inspecting completed W-minus-V results, especially the Llama contrast. It partitions the same planned comparisons into 50 IDs whose frozen original/repaired `raw_feedback` objects differ and 250 whose `raw_feedback` objects match. With two repetitions, the respective denominators are 100, 500 and 600 paired predictions per model. "Unchanged" refers only to `raw_feedback`; it does not mean the full verifier object, rendered feedback or prompt is unchanged. All 300 full verifier objects change version metadata.

The [released aggregate artifact](../results/extension_2026-10-10/posthoc_repair_localization.json) includes opaque group memberships, validity transitions, changes between two valid categorical outputs, joint gains/losses and subgroup contributions to the unchanged overall contrast. Each entry below is W joint matches minus V joint matches within its group; the two subgroup counts sum to the full result.

| Requested model | Net joint matches: 50 changed IDs | Net joint matches: 250 unchanged IDs | Net joint matches: all 300 IDs |
|---|---:|---:|---:|
| `gpt-6-astra` | -1 | 1 | 0 |
| `gpt-6.1-sol` | -1 | 0 | -1 |
| `gpt-6-luna` | 2 | 1 | 3 |
| `anthropic/claude-sonnet-5.5` | -4 | -3 | -7 |
| `anthropic/claude-haiku-5.5` | 1 | 17 | 18 |
| `google/gemini-3.8-flash` | -7 | -16 | -23 |
| `deepseek/deepseek-v4.1-flash` | -4 | 5 | 1 |
| `qwen/qwen3.8-flash` | -11 | -2 | -13 |
| `mistralai/mistral-large-4-0` | 0 | 0 | 0 |
| `meta-llama/llama-4-maverick` | 0 | 98 | 98 |
| `z-ai/glm-5.3-flash` | 0 | 2 | 2 |

Run `python -B scripts/replay_repair_localization.py --public-root .` to reconstruct every count from public categorical predictions and published group IDs. The group assignment itself cannot be independently reconstructed here because raw feedback is omitted; source hashes preserve its declared provenance. These are post hoc descriptive counts, not new confirmatory tests or evidence of a repair mechanism. Mixed batches, sampling variation and contextual effects prevent a causal interpretation. No main score, planned denominator or frozen confidence interval changes.
