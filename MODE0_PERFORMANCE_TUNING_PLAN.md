# MODE 0 Performance Tuning Plan

## Goal
Reduce MODE 0 end-to-end runtime while preserving output correctness and current sorting behavior.

## What is now instrumented
MODE 0 timing is now captured in make_video.py with a summary block:
- sql_select
- json_normalize
- hydrate_precomputed
- mongo_load
- unpickle_and_prep
- segment
- partition
- one_shot
- process_linear
- map_dispatch
- map_total

This provides a baseline before optimization work.

## Success criteria
- Minimum: 2x runtime reduction for representative MODE 0 production run.
- Target: 3x+ with no output regressions.
- Guardrail: Keep SORT_TYPE and NONE-object policy behavior unchanged.

## Phase 0: Baseline and hotspot ranking
1. Run one representative MODE 0 job.
2. Capture [MODE0 TIMING] summary.
3. Rank top bottlenecks by absolute seconds (not percent only).
4. Freeze this as baseline log.

Deliverable:
- Baseline timing report and top 3 hotspots.

## Phase 1: Low-risk wins (do first)
1. Reduce high-volume logging in hot loops with stricter debug gating.
2. Eliminate redundant conversions where possible in unpickle_and_prep.
3. Avoid repeated dataframe transformations if downstream does not need all fields.
4. Keep timing enabled and verify gains after each change.

Expected gain:
- 10% to 30%.

## Phase 2: Mongo and preprocessing optimization
1. Profile io.get_encodings_mongo usage and batch where possible.
2. Cache per-image fetched encoding blobs during a run if image IDs repeat.
3. Minimize repeated pickle/unpickle operations.
4. Use faster typed paths where arrays are already numpy-compatible.

Expected gain:
- 20% to 50% if mongo_load / unpickle_and_prep are dominant.

## Phase 3: Sorting-path optimization
1. Optimize prep_encodings_NN to avoid repeated expensive apply calls.
2. Reuse intermediate columns when one_shot_sort_dataframe followed by process_linear touches similar derived fields.
3. Review partition thresholds and KMeans invocation frequency.
4. Measure partition + one_shot + process_linear split cost separately.

Expected gain:
- 15% to 40% in sort-heavy workloads.

## Phase 4: Typed/Pickled MODE 0 output (high value, requested)
This is a key item and should be implemented.

Current pain:
- MODE 1 reparses CSV text columns into typed arrays/landmarks, repeating work done in MODE 0.

Plan:
1. In MODE 0, emit typed artifacts alongside CSV for each sorted output:
   - Preferred: parquet with pyarrow where feasible.
   - Fallback: pickle for complex object columns not parquet-friendly.
2. Include a manifest per output that records:
   - schema version
   - mode, cluster identifiers, sorting settings
   - file paths for csv + typed artifact
3. In MODE 1, prefer typed artifact if present; fallback to CSV for compatibility.
4. Keep dual-write for a transition period.

Expected gain:
- MODE 1 speedups (often large) and less conversion overhead.
- Cleaner end-to-end pipeline and easier reproducibility.

## Phase 5: Controlled parallelism (after hotspots are stable)
1. Parallelize independent MODE 0 units cautiously (topic/cluster chunks).
2. Limit worker count to avoid Mongo/DB and disk contention.
3. Keep deterministic file naming/output ordering.
4. Add per-worker timing for bottleneck attribution.

Expected gain:
- 1.5x to 4x depending on I/O contention.

## Validation checklist per phase
1. Runtime benchmark compared to baseline.
2. Same row counts per output CSV.
3. Spot-check output ordering stability.
4. No new runtime exceptions.
5. MODE 1 consumption still works (CSV fallback always available).

## Incremental execution order
1. Collect baseline from new MODE 0 timers.
2. Apply Phase 1 changes only.
3. Re-measure.
4. Apply Phase 2 if needed.
5. Re-measure.
6. Apply Phase 3.
7. Implement Phase 4 typed/pickled output.
8. Re-measure end-to-end MODE 0 + MODE 1.
9. Consider Phase 5 only if deadline still needs additional reduction.

## Notes
- Do not optimize blind: always use the MODE 0 timing summary to pick the next change.
- Typed/pickled artifact work is prioritized because it removes repeated parsing and prepares the pipeline for faster MODE 1 assembly.
