# MAKE_CACHE_MODE: Function-by-Function Migration Map

## Overview
This document maps **current code blocks** → **new function destinations** in a mechanical, patch-like format.
Source file: `make_video.py` (line numbers from make_video.py)

---

## PHASE 1: Path Resolution & Image Loading Helpers

### NEW FN 1: `resolve_row_io_paths(row)`
**Purpose:** Centralize path resolution logic (currently scattered in linear_test_df)

**Code to Extract:**
- `get_row_source_path_value(row)` nested fn [L2011-2020]
- `resolve_source_image_path(row)` nested fn [L2022-2032]  
- `get_cropped_cache_file(row)` nested fn [L2034-2056]
- `format_multiplier_for_cache_path()` nested fn [L2002-2009]

**Returns:**
```python
{
    "source_path": str,           # Resolved absolute path to source image
    "source_image_name": str,     # Basename of image
    "source_folder": str,         # Parent folder of source image
    "cropped_cache_file": str or None,  # Path to _cropped_* folder
    "inpaint_cache_file": str or None,  # Path to _inpaint_* folder
    "aspect_ratio_suffix": str    # e.g., "1.2_3.4"
}
```

**Thread-safe:** Yes (pure function, no shared state reads)

---

### NEW FN 2: `load_source_image_for_row(row, io_paths_dict)`
**Purpose:** Load and trim source image, with optional landmark drawing for debug

**Code to Extract:**
- Cache file existence check [L2375-2385] (skip if cache hit)
- `cv2.imread(open_path)` [L2390]
- Trim-bottom logic [L2395-2403]
- **NEW:** Optional `DRAW_TEST_LANDMARKS` debug behavior [L2398-2410]:
  - Draw face/hand landmarks for debug when enabled
  - This preserves the test-draw behavior you want to keep

**Inputs:**
- `row`: DataFrame row
- `io_paths_dict`: Result of `resolve_row_io_paths(row)`
- `DRAW_TEST_LANDMARKS`: Config flag
- `sort`: SortPose instance (for trim_bottom, draw_point)

**Returns:**
```python
(img, used_cached_cropped)  # img is np.ndarray or None, used_cached_cropped is bool
```

**Thread-safe:** Mostly (calls `sort.trim_bottom()` and `sort.draw_point()` — audit SortPose methods for thread-safety)

---

### NEW FN 3: `prepare_crop_context(img, row, face_landmarks, bbox)`
**Purpose:** Extract geometry and derived data for crop generation (pure output, no sort state mutations)

**Code to Extract:**
- Modified call to `sort.get_image_face_data(img, face_landmarks, bbox)` [~L2425]
  - **CRITICAL CHANGE:** Extract return values **only**, don't mutate sort state
  - Instead of: `sort.get_image_face_data(...)` mutating sort.image, sort.nose_2d, etc.
  - Do: Extract h, w, nose_2d, face_height **locally** and return in dict

**Implementation strategy:**
```python
def prepare_crop_context(img, row, face_landmarks, bbox):
    h, w = img.shape[:2]
    try:
        # Call sort method to compute nose_2d and face_height
        # but don't rely on sort.* mutations
        nose_2d, face_height = sort.get_image_face_data(img, face_landmarks, bbox)
        # Extract actual computed values (not sort instance state)
        return {
            "image": img,
            "h": h, "w": w,
            "face_landmarks": face_landmarks,
            "bbox": bbox,
            "nose_2d": nose_2d,
            "face_height": face_height,
            "is_valid": True
        }
    except Exception as e:
        print(f"prepare_crop_context failed: {e}")
        return {"is_valid": False, "image": img}
```

**Inputs:**
- `img`: np.ndarray
- `row`: DataFrame row (for yaw, roll, pitch if needed)
- `face_landmarks`: From row['face_landmarks']
- `bbox`: From row['bbox']

**Returns:** Context dict

**Thread-safe:** YES IF `sort.get_image_face_data()` is refactored to NOT mutate sort.* (see Phase 5: Thread-safety Audit)
**TEMPORARY:** If sort mutation happens, mark as audit point and use per-worker sort instance in Phase 5

---

## PHASE 2: Crop Generation (Replacing mixed compare_images logic)

### NEW FN 4: `generate_cropped_image(img, context)`
**Purpose:** Pure crop generation with explicit status return (no pair validation, no counter updates)

**Code to Extract from compare_images [L1752-1809]:**
- EXPAND path [L1768-1778]:
  - `sort.expand_image()` [L1768]
  - `sort.auto_edge_crop()` or `sort.crop_whitespace()` [L1770-1776]
- Direct crop path [L1779-1781]:
  - `sort.crop_image()`
- Status determination [L1783-1785]

**Inputs:**
- `img`: Source image (np.ndarray)
- `context`: Dict from `prepare_crop_context()` containing image geometry

**Returns:** Structured status tuple
```python
(cropped_image, status)
# where status in ["ok", "needs_inpaint", "failed"]
```

**Implementation:**
```python
def generate_cropped_image(img, context):
    if not context.get("is_valid"):
        return None, "failed"
    
    if sort.EXPAND:
        cropped_image, resize = sort.expand_image(img, context["face_landmarks"], context["bbox"])
        if AUTO_EDGE_CROP:
            cropped_image = sort.auto_edge_crop(context["bbox"], cropped_image, resize)
        else:
            cropped_image = sort.crop_whitespace(cropped_image)
        status = "ok"
    else:
        cropped_image, is_inpaint = sort.crop_image(img, context["face_landmarks"], context["bbox"])
        status = "needs_inpaint" if is_inpaint else "ok"
    
    if cropped_image is None:
        status = "failed"
    
    return cropped_image, status
```

**Thread-safe:** Mostly YES (calls SortPose crop helpers — audit in Phase 5)

---

## PHASE 3: Inpaint Refactor (Stop looping back to compare_images)

### NEW FN 5: `generate_cropped_via_inpaint(img, row, context)`
**Purpose:** Inpaint extension + crop on inpainted image, return pure crop without pair validation

**Code to Extract from in_out_paint [L2141-2285]:**
- Extension checking [L2153-2167]
- Inpaint file path resolution [L2168-2185]
- Cache file reuse check [L2186-2193]:
  - Load cached cropped if exists, but DON'T validate pair — return as-is for caller to validate
- Inpaint generation logic [L2194-2270]:
  - fetch_selfie_bbox
  - prepare_mask
  - extend_cv2
  - extend_lama
  - merge_inpaint
- **STOP at L2283:** Do NOT call compare_images to validate pair
- Instead: Return cropped image + status

**Inputs:**
- `img`: Source image
- `row`: DataFrame row (for image_id, bbox, etc.)
- `context`: Dict from prepare_crop_context (for nose_2d, face_height, etc.)

**Returns:**
```python
(cropped_image or cached_image, status, inpaint_context)
# status in ["ok", "cached", "needs_extend", "bailout"]
# inpaint_context for debugging/audit
```

**Critical change:** 
- Line L2283 `compare_images(...)` call is REMOVED
- Instead, return the inpainted image + context for caller to:
  1. Call `generate_cropped_image(inpaint_img, inpaint_context)` OR
  2. Call `validate_crop_pair(cropped_image, ...)` if pair validation is needed

**Thread-safe:** Mostly YES (calls sort.prepare_mask, sort.extend_lama — audit in Phase 5)

---

### NEW FN 6: `validate_crop_pair(cropped_image, last_image, current_image_id)`
**Purpose:** Run ONLY pair validation on a pre-cropped image, update counter_dict

**Code to Extract:**
- Entire `validate_cached_cropped()` nested function [L2057-2088] is a perfect template
- Move it out and rename to `validate_crop_pair()`
- Update to accept cropped_image directly (not via cache read)

**Inputs:**
- `cropped_image`: Already-cropped image to test
- `last_image`: Previous image for pair comparison
- `current_image_id`: For debugging

**Returns:**
```python
(validated_image, is_acceptable)
# validated_image is cropped_image if pair ok, else None
# is_acceptable is True if pair valid, False otherwise
```

**Side Effects:** Mutates `sort.counter_dict["first_run"]`, `sort.counter_dict["good_count"]`, `sort.counter_dict["isnot_face_count"]`

**Implementation:** Rename and extract [L2057-2088]

---

## PHASE 4: Orchestration — Two Parallel Paths

### NEW FN 7: `process_row_for_sequence(row, df_sorted, index)`
**Purpose:** Normal Mode 1 orchestrator — crops + validates pair + updates counters

**Code to Extract from linear_test_df main loop [L2356-2480]:**
- Path resolution [L2365-2368]
- Cache hit check [L2378-2388]
- Load source image [L2390-2410]
- Prepare crop context [~L2425]
- Generate crop [L2429]
- Inpaint fallback [L2452-2453]
- Save to output [L2458-2483]
- Counter updates [L2484-2485]

**Inputs:**
- `row`: DataFrame row
- `df_sorted`: Full DataFrame (for context)
- `index`: Row index

**Returns:**
```python
{
    "cropped_image": np.ndarray or None,
    "output_path": str,
    "cache_file": str,
    "status": str,  # "saved", "cached", "skipped"
    "skip_face": bool
}
```

**Orchestration logic:**
```
1. Resolve paths → resolve_row_io_paths()
2. Check cache hit & return if exists → load_source_image_for_row()
3. Load source image + trim → load_source_image_for_row()
4. Prepare context → prepare_crop_context()
5. Generate crop → generate_cropped_image()
6. If status == "needs_inpaint" → generate_cropped_via_inpaint()
7. Validate pair → validate_crop_pair()
8. Write outputs (cache + metas + output JPG)
9. Update sort.counter_dict["last_image"], ["last_image_id"]
```

**Thread-safe:** NO (mutates sort.counter_dict, sort state)

---

### NEW FN 8: `process_row_for_cache_only(row)`
**Purpose:** MAKE_CACHE_MODE orchestrator — crops only, skips validation + output writing

**Code to Extract from linear_test_df loop, SUBSET for cache-only:**
- Path resolution [L2365-2368]
- Cache skip check [L2378-2380] — **RETURN if cached**
- Load source image [L2390-2410]
- Prepare context [~L2425]
- Generate crop [L2429]
- Inpaint fallback [L2452-2453] (Version A: skip inpaint, just direct crop)
- Save ONLY to cache folder [L2461-2472]
- NO metas save, NO output JPG, NO counter updates

**Inputs:**
- `row`: DataFrame row

**Returns:**
```python
{
    "cache_file": str,
    "status": str,  # "cached", "generated", "failed"
    "cropped_image": np.ndarray or None,
    "skip_reason": str or None
}
```

**Orchestration logic (simplified):**
```
1. Resolve paths → resolve_row_io_paths()
2. Skip if cache exists → RETURN immediately
3. Load source image → load_source_image_for_row()
4. Prepare context → prepare_crop_context()
5. Generate crop → generate_cropped_image()
6. If status == "needs_inpaint" → EITHER:
   - Version A: Skip inpaint, try next row
   - Version B: Call generate_cropped_via_inpaint() (later)
7. Write ONLY to cache folder (no validation, no metas, no output)
8. NO counter updates
```

**Thread-safe:** YES (only reads row data, writes cache files independently, no shared state mutations)

---

## PHASE 5: Thread-Safety Audit Points

### Audit Location 1: `sort.get_image_face_data()` in mp_sort_pose.py [L2396]
**Current behavior:** Mutates instance state
```python
self.image = image
self.h, self.w = ...
self.nose_2d = ...
self.face_height = ...
```

**Audit task:**
- [ ] Refactor to return (nose_2d, face_height) without mutating self.*
- [ ] OR: Modify `prepare_crop_context()` to use per-worker sort instance
- [ ] OR: Add thread-lock around get_image_face_data calls

---

### Audit Location 2: `sort.crop_image()`, `sort.expand_image()`, `sort.auto_edge_crop()`
**Question:** Do these read/write shared sort.* state, or just use passed parameters?

**Audit task:**
- [ ] Check if they access sort.image, sort.nose_2d, sort.face_height
- [ ] If yes: refactor to accept explicit parameters
- [ ] If no: safe to call from multiple threads

---

### Audit Location 3: `shift_bbox()` in in_out_paint [~L2275]
**Current behavior:** Calls `sort.prepare_mask()` which may mutate sort.nose_2d

**Code:**
```python
bbox = shift_bbox(row['bbox'], extension_pixels)
```

**Audit task:**
- [ ] Verify shift_bbox doesn't mutate shared state
- [ ] Check sort.prepare_mask for state mutations

---

### Audit Location 4: `sort.test_or_lookup_face_pair()`
**Question:** Does this read/write shared caches or state?

**Audit task:**
- [ ] Check for in-memory caches
- [ ] Check for database session state
- [ ] Verify thread-safe under concurrent calls

---

## PHASE 6: Threading Integration

### NEW FN 9: `process_csv_cache_only(df_sorted, csv_path)`
**Purpose:** Thread pool orchestrator for MAKE_CACHE_MODE

**Inputs:**
- `df_sorted`: DataFrame from CSV
- `csv_path`: CSV file path (for logging)

**Orchestration:**
```python
def process_csv_cache_only(df_sorted, csv_path):
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [
            executor.submit(process_row_for_cache_only, row)
            for _, row in df_sorted.iterrows()
        ]
        results = [f.result() for f in futures]
    return results
```

**Thread-safe:** YES (each thread calls process_row_for_cache_only which is thread-safe)

---

## PHASE 7: Branch Point in save_images_from_csv_folder

### Location: make_video.py save_images_from_csv_folder() [L3337]

**Current flow:**
```
load_canonical_multiplier_registry_once()
  ↓
LOOP CSV files:
  load df_sorted from CSV
    ↓
  set_my_counter_dict()
    ↓
  set_multiplier_and_dims()
    ↓
  recheck_is_dupe_of()  ← DEDUPE LOGIC
    ↓
  remove_duplicates()
    ↓
  linear_test_df(df_sorted)  ← SEQUENTIAL PROCESSING
```

**New flow with MAKE_CACHE_MODE:**
```
load_canonical_multiplier_registry_once()
  ↓
LOOP CSV files:
  load df_sorted from CSV
    ↓
  set_my_counter_dict()
    ↓
  set_multiplier_and_dims()
    ↓
  IF MAKE_CACHE_MODE:          ← NEW BRANCH
    cache_only_df(df_sorted)    ← THREADED CACHE GENERATION
  ELSE:                          ← EXISTING PATH
    recheck_is_dupe_of()
    remove_duplicates()
    linear_test_df(df_sorted)
```

**Code change location:** ~L3400 in save_images_from_csv_folder
```python
def save_images_from_csv_folder():
    load_canonical_multiplier_registry_once()
    
    for csv_path in CSV_files:
        df_sorted = load_df_sorted_from_csv(csv_path)
        set_my_counter_dict()
        set_multiplier_and_dims(df_sorted, cluster_no, pose_no)
        
        if MAKE_CACHE_MODE:
            # Branch: cache-only mode (threaded)
            cache_results = process_csv_cache_only(df_sorted, csv_path)
            # Optional: write summary of cache generation
        else:
            # Original path: normal Mode 1 assembly
            recheck_is_dupe_of()
            remove_duplicates()
            linear_test_df(df_sorted)
```

**Rationale:** 
- MAKE_CACHE_MODE skips dedupe (not needed for cache)
- Direct to parallel cache generation
- Normal path unchanged (preserves Mode 1 semantics)

---

## Summary: Function Extraction Order (Recommended)

**Phase 1 (Helpers):**
1. Extract `resolve_row_io_paths()` ✓
2. Extract `load_source_image_for_row()` ✓
3. Extract `prepare_crop_context()` ✓

**Phase 2 (Core Crop):**
4. Extract `generate_cropped_image()` ✓

**Phase 3 (Inpaint):**
5. Extract `generate_cropped_via_inpaint()` ✓
6. Extract `validate_crop_pair()` (rename from validate_cached_cropped) ✓

**Phase 4 (Orchestration):**
7. Extract `process_row_for_sequence()` ✓
8. Extract `process_row_for_cache_only()` ✓

**Phase 5 (Audit):**
9. Audit `sort.get_image_face_data()`, crop helpers, shift_bbox, test_or_lookup_face_pair

**Phase 6 (Threading):**
10. Create `process_csv_cache_only()` with ThreadPoolExecutor

**Phase 7 (Integration):**
11. Add MAKE_CACHE_MODE branch in `save_images_from_csv_folder()`

---

## Notes

- **DRAW_TEST_LANDMARKS debug behavior:** Preserved in `load_source_image_for_row()` [L2398-2410]
- **Version A inpaint:** Direct crop only, skip inpaint fallback for cache mode initially
- **Version B inpaint:** With fallback to `generate_cropped_via_inpaint()` — implement after Phase 5 audit
- **Thread-safety:** Depends on Phase 5 audit results; current design assumes SortPose methods are mostly pure
