# Production Pipeline Steps

## 2026 current state of affairs

At present, in Fall 2026, I am building out the main chapters, in groups of topics. 

- The Office, T11, T37 (money objects) for Basel
    - Currently being polished for Thoma
    - Looping videos are done
    - signatures, arms done. body IP. redo sig with EXISTS
- The Gym, T0, 3, 15, 45 for LA
    - Needs a repolish. 
    - Fine tune the canonical multipliers for best dimensions
    - Make looping? 
    - signatures placed, not assigned
- The Store, T43, 59 
    - Just starting
    - Assigned BODY/ARMS clusters
    - signatures IP


## Steps you need to do

There is a bunch of preliminary stuff, and I'm not sure how much needs to be double checked

0. Do I need to check whether all new seg_big have topics? (done SegmentBig Oct 8 2026)
1. normalize all body landmarks (done SegmentBig Oct 8 2026)
2. normalize all obj bbox (done SegmentBig Oct 8 2026 -- will need to be redone on all new detections, query max detection with norm to start from)
3. Get INCLUDE_LEG_POSE_FEATURES for arms with utilities/extract_hand_feet_locations.py (done Encodings Oct 8 2026)
4. Assign objectsignatures. Full detailed info is in Clustering_SQL.py
    1. Assign objects to positions - analysis/imagesdetections_debug/object_placement_audit/rerun_imagesdetections_assignments.py
    2. Assign those positions to signatures - Clustering_SQL.py
5. Assign clusters.
    0. run utilities/detect_is_feet.py (done SegmentBig Oct 8 2026)
    1. Body 
    2. Arms
    3. if needed Hands Pose and Hands Gesture
6. If needed, calculate background color with calculate_background_color.py (only super relevant for Looping split color and Paris Photo style videos I think?) 
7. Query all fusion clusters to generate counts/policy for each cluster
8. Build Make_Video.py