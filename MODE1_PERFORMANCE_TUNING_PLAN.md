# MODE 1 Performance Tuning Plan

## Objective
Reduce total MODE 1 runtime enough to finish full production runs within deadline, while preserving current behavior and output quality.

## Scope
This plan focuses on MODE 1 in make_video.py (assemble from CSV folder) and adjacent hot paths in mp_sort_pose.py / image processing helpers.

## Success Criteria
- Primary: End-to-end MODE 1 wall time reduced by at least 2x.
- Stretch: 3x to 5x reduction.
- Correctness: Same output ordering and no regressions in visual quality unless explicitly toggled to fast preview mode.

## Constraints
- Keep the current SORT_TYPE behavior and NONE-object routing behavior intact.
- Avoid high-risk refactors until profiling confirms bottlenecks.
- Implement incrementally with rollback points after each phase.

## Baseline and Measurement First
Before changing logic, collect a reproducible baseline.

### Baseline Metrics to Record
- Total runtime for one representative full MODE 1 run.
- Per-stage timing:
  - CSV read
  - dataframe parse/convert
  - DB dedupe/recheck queries
  - image load/decode
  - crop/transform/inpaint/upscale
  - final write/composite
- Throughput:
  - images/sec overall
  - images/sec during assembly stage only
- Resource profile:
  - CPU utilization
  - RAM peak
  - disk read throughput
  - DB query count and total DB time

### Required Instrumentation
Add lightweight timers around major blocks in MODE 1 path in main() and save_images_from_csv_folder().
Log per-file and aggregate timing summary at end of run.

## Bottleneck Hypotheses (ordered by likely impact)
1. Repeated parsing/deserialization from CSV in MODE 1.
2. Image IO + decode + resize/crop costs dominate.
3. Many small DB calls (dedupe / metadata checks).
4. Python row-wise apply/eval overhead in dataframe conversion.
5. Verbose logging in tight loops.

## Phased Implementation Plan

## Phase 0: Instrumentation and Baseline (Low risk, mandatory)
### Tasks
1. Add a tiny timing utility (context manager or helper function).
2. Wrap MODE 1 stages with timers.
3. Emit one consolidated summary report at run end.
4. Capture one baseline run and save output in a timestamped log.

### Exit Criteria
- Timing report clearly identifies top 2 bottlenecks by wall time.

### Rollback
- Disable timing with a single flag.

---

## Phase 1: Easy Wins (Low risk, fast turnaround)
### Tasks
1. Reduce or gate high-volume print calls behind strict debug flag.
2. Replace repeated per-column applymap/row lambdas with vectorized operations where safe.
3. Avoid redoing invariant setup per CSV file (precompute static maps once).
4. Ensure CSV reads use efficient engine/options for large files.

### Expected Gain
10% to 30%.

### Validation
- Runtime delta on same dataset.
- Output row counts and ordering unchanged.

---

## Phase 2: Database Round-Trip Reduction (Medium impact, low/medium risk)
### Tasks
1. Precollect all image_id values from all df_sorted files first.
2. Run batched DB fetches once (or in large chunks) instead of per-file/per-loop queries.
3. Build in-memory lookup dictionaries for dedupe decisions.
4. Reuse these dictionaries throughout MODE 1.

### Expected Gain
10% to 40%, depending on DB latency share.

### Validation
- Query count drops significantly.
- Dedupe behavior matches previous run.

---

## Phase 3: Image IO and Transform Acceleration (High impact)
### Tasks
1. Add bounded LRU cache for decoded images and/or frequent metadata.
2. Ensure repeated references to same image do not trigger repeated disk decode.
3. Add optional FAST_PREVIEW toggles:
   - disable expensive inpaint/upscale
   - lower resize quality
4. Keep default full-quality path unchanged.

### Expected Gain
20% to 60% in full quality; more in preview mode.

### Validation
- Compare cache hit rate.
- Confirm full-quality mode output parity.

---

## Phase 4: Reduce MODE 1 Parse Cost with Typed Intermediate (High impact)
### Strategy
Shift from text-heavy CSV-only interchange to typed intermediate artifacts.

### Tasks
1. In MODE 0, optionally write typed intermediate (parquet/pickle) alongside CSV.
2. In MODE 1, prefer typed intermediate when available.
3. Keep CSV fallback for backward compatibility.
4. Remove repeated eval/string-to-list conversions when typed data is present.

### Expected Gain
1.5x to 3x end-to-end in data-heavy runs.

### Validation
- Row-level equivalence checks between CSV path and typed path.
- End-to-end ordering and output parity.

---

## Phase 5: Parallelize Independent Work Units (Potentially largest gain)
### Tasks
1. Parallelize per-CSV preprocess/assembly with process pool (not threads for CPU-heavy sections).
2. Keep final write ordering deterministic (serialize commit/write phase if required).
3. Tune worker count to avoid IO thrash (start with min(physical_cores, 4)).
4. Add watchdog logging for worker failures/timeouts.

### Expected Gain
2x to 6x for CPU-heavy workloads, bounded by disk/DB contention.

### Validation
- Deterministic outputs across repeated runs.
- No dropped/duplicated frames.

---

## Work Plan by Priority and Deadline
## Week 1 / Immediate
1. Phase 0 instrumentation baseline.
2. Phase 1 easy wins.
3. Phase 2 DB batching.

## Week 2 / If needed for deadline
1. Phase 3 image cache and fast preview mode.
2. Phase 4 typed intermediate.

## Week 3 / Optional stretch
1. Phase 5 parallelization.

## Risk Register
1. Behavior drift from aggressive optimization.
   - Mitigation: phase gates + output parity checks.
2. Memory growth from caches.
   - Mitigation: bounded LRU, configurable limits.
3. Disk contention with parallel workers.
   - Mitigation: conservative worker count, staged IO.
4. Difficult rollback after deep refactor.
   - Mitigation: feature flags and branch-per-phase.

## Validation Checklist per Phase
- Runtime benchmark against same fixed sample.
- Output count parity.
- Ordering parity (where required).
- Visual spot-check of random sample.
- Log scan for warnings/errors.

## Suggested Feature Flags
- ENABLE_MODE1_TIMING
- MODE1_FAST_PREVIEW
- MODE1_USE_TYPED_INTERMEDIATE
- MODE1_PARALLEL_WORKERS
- MODE1_IMAGE_CACHE_SIZE

## Deliverables
1. Timing report template and baseline logs.
2. Incremental PR/commit per phase.
3. Final comparison table:
   - baseline runtime
   - runtime after each phase
   - cumulative speedup

## First Execution Session (recommended)
1. Add timing instrumentation only.
2. Run one representative MODE 1 job.
3. Rank top bottlenecks by time share.
4. Implement only top one in next session.

This plan is intentionally incremental so we can stop at any point once runtime is good enough for the deadline.
