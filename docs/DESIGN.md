# Study design and interpretation

The source repository contains 40,576 extracted verse pairs, not 40,576 poems. We sampled 300 evaluation pairs balanced across M/F2/F3 (100 each), with a separate 60-pair example pool (20 each). M, F2 and F3 indicate final, penultimate and antepenultimate stress. Reference labels are stored automatic corpus classifications; there was no human gold annotation.

Pair normalization, exact/near-duplicate filtering and the original 40 benchmark pairs plus an earlier 100-pair pilot define the exclusions. No reliable poem identifiers were available, so this is not a claim of poem-disjoint sampling. The positive, balanced sample does not evaluate rhyme detection or estimate corpus prevalence. Source acquisition and text selection are documented in the original preparation code and historical data contract.

## Conditions and accounting

- **S:** compact common taxonomy, one call.
- **C:** S plus the complete recovered selectable text of the 19-page theoretical source, one call. Raster-only figures/examples and original typography are omitted; tables are linearized. This is full-context input, not a measured caching-speed experiment.
- **E:** S plus 15 seeded examples, five per stress class, from the separate 60-pair pool. Each repetition uses a different fixed example draw.
- **R:** reconsider the saved S answer without verifier feedback.
- **V:** revise the identical saved S answer with the original verifier's precomputed feedback.
- **W:** revise the identical saved S answer with the isolated engineering repair's feedback.

R/V/W share the same S answer within a model/batch/repetition, so changes are paired. There are 30 main batches of 10 pairs, two repetitions and three models. Main predictions total 10,800. The separate legacy40 condition uses S only and contributes 240 predictions. Planned physical jobs total 1,104, with 1,110 retained attempts and zero invalid final predictions in the recorded study.

The acceptance rule required successful process exit plus a completed, schema-valid, tool-free response. Six earlier retained answers passed the latter checks but failed the exit gate and were retried. The private post-hoc first-completed-valid-answer sensitivity changed 7/5,400 S/C/E predictions, with a maximum cell-metric shift of 0.50 percentage points; it did not change the primary results or construct revision counterfactuals. That raw-evidence diagnostic cannot be rerun from this public release. Aggregate usage includes all retained attempts.

## Metrics and uncertainty

Primary **joint agreement** requires the exact stress class and full canonical feature set. Feature order does not matter. The prespecified **core joint** metric removes COPY/MOSAIC from both sides and otherwise applies the same rule. Stress, feature-exact, micro feature F1, per-item F1, class breakdown and repeat stability are secondary measurements.

Invalid/failed outputs remain in the planned denominator. Component feature scores compare sets regardless of the predicted stress type, including NONE; they are not detection metrics. Invalid outputs contribute all reference features as false negatives. Valid empty/empty sets score item F1=1; invalid outputs score 0. The original code and synthetic tests specify the complete contract.

Two repetitions are averaged within each item before a paired cluster bootstrap over the 30 fixed inference batches. Seed: 20261008; resamples: 10,000. A source-group sensitivity resamples the observed groups (11 in main300), not inferred poem identities. Intervals are descriptive 95% percentile intervals without multiplicity adjustment; no significance declarations are made. E repeat stability combines example-selection and decoding variability.

## What the results support

Stress agreement is generally high, while complete feature-set agreement is more sensitive to prompt and feedback conditions. The reference convention sometimes omits COPY/MOSAIC features returned by the live verifier. IDV and PURE co-occurrence also expose differences between the compact instruction, theory descriptions and inherited implementation. These provide hypotheses for the C-condition drop, not a demonstrated causal explanation or a new linguistic gold standard.

The original and repaired verifiers are related to the system that supplied the reference labels. Feedback gains therefore indicate improved agreement with those conventions; they are not independent evidence of improved linguistic truth. The repair addresses representation/interface consistency, not a validated redesign of Greek pronunciation or rhyme theory.

The three requested models belong to one provider family. The aliases, high reasoning setting and observed usage are recorded, but backend weights are not independently attested. Recorded job duration sums retained attempts and includes process/connectivity delays; it is not isolated model compute latency. Cached/reasoning token counters are components, not additional tokens to add to their enclosing totals. No dollar cost is inferred from subscription usage.
