# MySQL Connection Timeout Fix - Implementation Plan

## Problem
MySQL connections are timing out due to stale connections in the connection pool.

## Solution
Add connection pool configuration to all `create_engine()` calls:
- `pool_pre_ping=True` - Tests connections before using them
- `pool_recycle=600` - Recycles connections after 1 hour (3600 seconds)

## Files to Update (29 files)

### Main Scripts (9 files)
1. `Clustering_SQL.py` - 2 instances (lines 342, 346)
2. `process_placards.py` - 1 instance (line 32)
3. `calc_phone_bbox.py` - 2 instances (lines 34, 38)
4. `calculate_background_color.py` - 2 instances (lines 73, 77)
5. `topic_model.py` - 2 instances (lines 149, 153)
6. `fetch_bagofkeywords.py` - 1 instance (line 31)
7. `export_all_images.py` - 2 instances (lines 584, 588)
8. `detect_multiple_faces.py` - 2 instances (lines 453, 459)
9. `mask_detect_hsv.py` - 1 instance (line 25)

### Processing Scripts (6 files)
10. `tench_bbox_calc_testing.py` - 2 instances (lines 824, 828)
11. `copy_toSSD_files.py` - 2 instances (lines 98, 102)
12. `normalize_lms.py` - 2 instances (lines 111, 115)
13. `ingest_jsonl.py` - 1 instance (line 290)
14. `fetch_segment_keywords.py` - 2 instances (lines 70, 74)
15. `TSPsort_classTEST.py` - (imported but not instantiated in visible code)

### Utilities (14 files)
16. `utilities/move_encodings_mongo.py` - 1 instance (line 30)
17. `utilities/update_compare_table_NULLs.py` - 2 instances (lines 35, 238)
18. `utilities/insert_segment.py` - 1 instance (line 32)
19. `utilities/update_segment.py` - 2 instances (lines 32, 73)
20. `utilities/detect_which_hand_visible.py` - 2 instances (lines 54, 174)
21. `utilities/detect_is_feet.py` - 1 instance (line 23)
22. `utilities/deshard_mysql.py` - 1 instance (line 46)
23. `utilities/compare_sql_mongo_encodings.py` - 2 instances (lines 33, 153)
24. `utilities/update_mongo_NULLs.py` - 1 instance (line 31)
25. `utilities/deshard_from_json.py` - 1 instance (line 65)
26. `utilities/cleanup_missing_face_embeddings.py` - 1 instance (line 36)
27. `utilities/repickle_encodings.py` - 1 instance (line 26)
28. `utilities/move_tokens_mongo.py` - 1 instance (line 30)
29. `utilities/assign_h_w.py` - 1 instance (line 30)
30. `utilities/add_to_segment.py` - 1 instance (line 49)
31. `utilities/extract_hand_feet_locations.py` - 1 instance (line 42)
32. `utilities/update_imagename.py` - 1 instance (line 35)
33. `utilities/find_diff_sql_mongo_encodings.py` - 2 instances (lines 33, 135)
34. `utilities/project_landmarks.py` - 2 instances (lines 81, 85)
35. `utilities/recalculate_xyz_rad.py` - 1 instance (line 30)
36. `utilities/salvage_cluster_medians.py` - 2 instances (lines 121, 125)
37. `utilities/delete_rows.py` - 1 instance (line 27)
38. `utilities/validate_sql_mongo_encodings.py` - 2 instances (lines 36, 704)
39. `utilities/update_NULL_encodings.py` - 1 instance (line 29)

## Code Change Examples

### Pattern 1: With poolclass=NullPool (most common)

**BEFORE:**
```python
engine = create_engine("mysql+pymysql://{user}:{pw}@/{db}?unix_socket={socket}".format(
    user=db['user'], pw=db['pass'], db=db['name'], socket=db['unix_socket']
), poolclass=NullPool)
```

**AFTER:**
```python
engine = create_engine("mysql+pymysql://{user}:{pw}@/{db}?unix_socket={socket}".format(
    user=db['user'], pw=db['pass'], db=db['name'], socket=db['unix_socket']
), pool_pre_ping=True, pool_recycle=600, poolclass=NullPool)
```

### Pattern 2: Without poolclass (less common)

**BEFORE:**
```python
engine = create_engine("mysql+pymysql://{user}:{pw}@/{db}?unix_socket={socket}".format(
    user=db['user'], pw=db['pass'], db=db['name'], socket=db['unix_socket']
))
```

**AFTER:**
```python
engine = create_engine("mysql+pymysql://{user}:{pw}@/{db}?unix_socket={socket}".format(
    user=db['user'], pw=db['pass'], db=db['name'], socket=db['unix_socket']
), pool_pre_ping=True, pool_recycle=600)
```

### Pattern 3: F-string format (newer style)

**BEFORE:**
```python
engine = create_engine(
    f"mysql+pymysql://{db['user']}:{db['pass']}@/{db['name']}?unix_socket={db['unix_socket']}",
    poolclass=NullPool
)
```

**AFTER:**
```python
engine = create_engine(
    f"mysql+pymysql://{db['user']}:{db['pass']}@/{db['name']}?unix_socket={db['unix_socket']}",
    pool_pre_ping=True,
    pool_recycle=600,
    poolclass=NullPool
)
```

### Pattern 4: Conditional (if/else for unix_socket vs host)

**BEFORE:**
```python
if db['unix_socket']:
    engine = create_engine("mysql+pymysql://{user}:{pw}@/{db}?unix_socket={socket}".format(
        user=db['user'], pw=db['pass'], db=db['name'], socket=db['unix_socket']
    ), poolclass=NullPool)
else:
    engine = create_engine("mysql+pymysql://{user}:{pw}@{host}/{db}"
                                .format(host=db['host'], db=db['name'], user=db['user'], pw=db['pass']), poolclass=NullPool)
```

**AFTER:**
```python
if db['unix_socket']:
    engine = create_engine("mysql+pymysql://{user}:{pw}@/{db}?unix_socket={socket}".format(
        user=db['user'], pw=db['pass'], db=db['name'], socket=db['unix_socket']
    ), pool_pre_ping=True, pool_recycle=600, poolclass=NullPool)
else:
    engine = create_engine("mysql+pymysql://{user}:{pw}@{host}/{db}"
                                .format(host=db['host'], db=db['name'], user=db['user'], pw=db['pass']), 
                           pool_pre_ping=True, pool_recycle=600, poolclass=NullPool)
```

## Implementation Steps

1. **Test on one file first** - Start with a high-usage file like `Clustering_SQL.py`
2. **Verify it works** - Run the script and confirm no connection timeout errors
3. **Apply to all files** - Use multi-file search/replace or script to update all instances
4. **Test critical scripts** - Run tests on main processing scripts to ensure stability
5. **Monitor** - Watch for any connection-related errors in production

## Notes

- `pool_pre_ping=True` adds a small overhead (lightweight SELECT 1 query) but prevents stale connections
- `pool_recycle=600` (1 hour) is conservative; can be adjusted based on MySQL's `wait_timeout` setting
- Most scripts use `poolclass=NullPool` which disables pooling entirely - the new parameters will still help with connection health
- Some scripts create multiple engines (e.g., `update_compare_table_NULLs.py` has `engine` and `thread_engine`) - update all instances

## Testing Checklist

- [ ] Update `Clustering_SQL.py` (high priority - 2 instances)
- [ ] Update `process_placards.py` (high priority - 1 instance)
- [ ] Update `detect_multiple_faces.py` (high priority - 2 instances)
- [ ] Test one script with heavy DB usage
- [ ] Apply changes to remaining 36 files
- [ ] Run integration tests
- [ ] Monitor logs for connection errors
