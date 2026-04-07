import os
import pickle
import cv2
import numpy as np
import sqlalchemy
from sqlalchemy import Float, create_engine, Column, Integer, Boolean, String
from sqlalchemy.orm import sessionmaker, aliased
from sqlalchemy.pool import NullPool
from pathlib import Path
from pymediainfo import MediaInfo

# importing project-specific models
import sys
ROOT_GITHUB = os.path.join(Path.home(), "Documents/GitHub/takingstock/")
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, ROOT_GITHUB)

from my_declarative_base import SegmentBig, SegmentTable, Encodings, Base, Images
from mp_pose_est import SelectPose

# MongoDB setup
import pymongo
mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
mongo_db = mongo_client["stock"]
mongo_collection = mongo_db["encodings"]  # adjust collection name if needed

VERBOSE = True
SSD=False
SSD_FOLDER_OVERRIDE = "/Volumes/OWC54/segment_images_40xDetections" # set to None to use default SSD path from config
HelperTable_name = "SegmentHelper_oct2025_evens_quarters" # if you set to None, comment out the helpertable join in the query
# HelperTable_name = None # if you set to none, comment out the helpertable join in the query
REQUIRE_IMAGES_H_W = False # if True, will skip rows where Images.h or Images.w is NULL, since we need those for the blank image creation

# MySQL setup (preserving credentials framework)
from mp_db_io import DataIO
io = DataIO(SSD,VERBOSE,SSD_FOLDER_OVERRIDE)
db = io.db
engine = create_engine(
    f"mysql+pymysql://{db['user']}:{db['pass']}@/{db['name']}?unix_socket={db['unix_socket']}",
    pool_pre_ping=True,
    pool_recycle=600,
    poolclass=NullPool
)
Session = sessionmaker(bind=engine)
session = Session()

# Batch processing parameters
batch_size = 1000
last_id = 0 # this is now image_id, not encoding_id. use this to restart the process midway

class HelperTable(Base):
    __tablename__ = HelperTable_name
    seg_image_id=Column(Integer,primary_key=True, autoincrement=True)
    image_id = Column(Integer, primary_key=True, autoincrement=True)
class ImagesArmsPoses3D(Base):
    __tablename__ = "ImagesArmsPoses3D"
    image_id = Column(Integer, primary_key=True)
    cluster_id = Column(Integer)
    cluster_dist = Column(Float)

print(f"Starting batch processing with batch_size={batch_size}, last_id={last_id}, REQUIRE_IMAGES_H_W={REQUIRE_IMAGES_H_W}")
print(f"HelperTable_name: {HelperTable_name}, SSD_FOLDER_OVERRIDE: {SSD_FOLDER_OVERRIDE}")

def get_shape(site_name_id, imagename):
    site_specific_root_folder = io.folder_list[site_name_id]
    file_path = site_specific_root_folder + "/" + imagename

    try:
        if io.platform == "darwin":
            media_info = MediaInfo.parse(file_path, library_file="/opt/homebrew/Cellar/libmediainfo/26.01/lib/libmediainfo.dylib")
        else:
            media_info = MediaInfo.parse(file_path)
    except Exception as e:
        print(f"🚨 MediaInfo read failed for image file: {file_path} (site_name_id={site_name_id}, imagename={imagename})")
        print(f"   Error: {type(e).__name__}: {e}")
        return None, None

    for track in media_info.tracks:
        if track.track_type == 'Image':
            return track.height, track.width

    print(f"🚨 MediaInfo found no Image track: {file_path} (site_name_id={site_name_id}, imagename={imagename})")
    return None, None

def debug_face_pose_simple(self, bbox, faceLms, image_id):
    """Simplified debug with corrected coordinate system"""

    # Extract landmarks
    faceXY, image_points = self.extract_key_lms_for_faceXYZ(bbox, faceLms)
    
    # Solve
    success, r_vec, t_vec, method = self.solve_head_pose_robust(image_points)
    
    if success:
        self.r_vec = r_vec
        self.t_vec = t_vec
        angles = self.rotationMatrixToEulerDegrees()
        eye_roll = self.get_roll_from_landmarks(faceLms)
        
        if VERBOSE:
            print(f"Image ID: {image_id} Method: {method}")
            print(f"Pitch: {angles[0]:6.2f}° (neg=looking down, pos=looking up)")
            print(f"Yaw:   {angles[1]:6.2f}° (neg=looking left, pos=looking right)")
            print(f"Roll:  {angles[2]:6.2f}° (PnP) vs {eye_roll:6.2f}° (eyes)")
        
        # Sanity check warnings
        if abs(angles[0]) > 90:
            print("  ⚠ WARNING: Pitch > 90° suggests coordinate system issue")
        if abs(angles[1]) > 90:
            print("  ⚠ WARNING: Yaw > 90° suggests coordinate system issue")
    else:
        print("❌ FAILED to solve pose")
    
    return success


while True:
    s = aliased(HelperTable, name='s')
    # ihp = aliased(ImagesArmsPoses3D, name='ihp')
    query = (
        session.query(
            Encodings.encoding_id,
            Encodings.image_id,
            Encodings.face_x,
            Encodings.face_y,
            Encodings.face_z,
            Encodings.bbox,
            Encodings.is_feet,
            Images.h,
            Images.w,
            Images.site_name_id,
            Images.imagename,
        )
        .join(s, s.image_id == Encodings.image_id)
        # .join(ihp, s.image_id == ihp.image_id)
        .join(Images, Images.image_id == Encodings.image_id)
        .filter(
            Encodings.mongo_face_landmarks.is_(True),
            Encodings.image_id > last_id,
            Encodings.pitch.is_(None),
        )
    )

    if REQUIRE_IMAGES_H_W:
        query = query.filter(
            Images.h.isnot(None),
            Images.w.isnot(None),
        )

    results = query.limit(batch_size).all()
    # .order_by(Encodings.image_id)
        # Encodings.is_feet.is_(None),

    if not results:
        print("No more rows to process. Exiting.")
        break
    
    print(f"Processing {len(results)} records...")

    batch_stats = {
        'queried': len(results),
        'h_w_filled': 0,
        'h_w_fill_failed': 0,
        'pose_saved': 0,
        'skipped': 0,
    }

    # Process in batches
    for i in range(0, len(results), batch_size):
        batch = results[i:i + batch_size]
        batch_success = []
        images_hw_updates = []
        for encoding_id, image_id, face_x, face_y, face_z, bbox, is_feet, h, w, site_name_id, imagename in batch:
            if face_x is None or face_y is None or face_z is None:
                print(f"Skipping image_id {image_id} due to missing face_xyz data.")
                batch_stats['skipped'] += 1
                continue
            if VERBOSE:
                print(f"Processing image_id: {image_id}, encoding_id: {encoding_id}")

            valid_h_w = isinstance(h, int) and isinstance(w, int) and h > 0 and w > 0
            if not valid_h_w:
                if REQUIRE_IMAGES_H_W:
                    print(f"Skipping image_id {image_id}: invalid image dimensions h={h}, w={w}")
                    batch_stats['skipped'] += 1
                    continue

                recovered_h, recovered_w = get_shape(site_name_id, imagename)
                if recovered_h is None or recovered_w is None:
                    print(f"🚨 ALERT image_id {image_id}: missing/invalid h,w and could not recover from file. site_name_id={site_name_id}, imagename={imagename}")
                    batch_stats['h_w_fill_failed'] += 1
                    batch_stats['skipped'] += 1
                    continue

                h, w = int(recovered_h), int(recovered_w)
                images_hw_updates.append({
                    'image_id': int(image_id),
                    'h': h,
                    'w': w,
                })
                batch_stats['h_w_filled'] += 1
                if VERBOSE:
                    print(f"{image_id} - Recovered and queued h,w update: h={h}, w={w}")

            # 2. Fetch pickled face_landmarks from Mongo
            mongo_doc = mongo_collection.find_one({"image_id": image_id}, {"face_landmarks": 1})
            is_feet = False
            faceLms = None
            if mongo_doc and mongo_doc.get("face_landmarks"):
                # 3. Unpickle to MediaPipe object
                faceLms = pickle.loads(mongo_doc["face_landmarks"])
            else:
                print(f"No face_landmarks found in MongoDB for image_id {image_id}. Skipping.")
                batch_stats['skipped'] += 1
                continue

            # create a blank (all black) image of dimensions h x w
            # validate h and w (can be recovered above if REQUIRE_IMAGES_H_W is False)
            if not (isinstance(h, int) and isinstance(w, int) and h > 0 and w > 0):
                print(f"Skipping image_id {image_id}: invalid image dimensions h={h}, w={w}")
                batch_stats['skipped'] += 1
                continue
            blank_image = np.zeros((h, w, 3), np.uint8)

            # 3. Use SelectPose to estimate pose from blank image
            pose = SelectPose(blank_image)

            # print(f"Calculating pose for image_id {image_id}..., ", bbox)
            if bbox is None: continue

            try:
                bbox = io.unstring_json(bbox)
            except Exception as e:
                print(f"Error unstringing JSON for image_id {image_id}: {e}")
                continue
            
            # check to see if faceLms has the expected structure
            if not hasattr(faceLms, 'landmark') or len(faceLms.landmark) == 0:
                print(f"Invalid faceLms structure for image_id {image_id}. Skipping this data: {faceLms}.")
                continue
            pose.calc_face_data(faceLms)

            if VERBOSE: print(f"{image_id} - Calculating pose with SelectPose...")
            # Initialize once
            pose.model_points = pose.get_model_points_corrected()

            # Then for each face:
            result = pose.calculate_face_pose_final(bbox, faceLms)
            if result:
                if VERBOSE: print(f"{image_id} - Pitch: {result['pitch']:.1f}°, Yaw: {result['yaw']:.1f}°, Roll: {result['roll']:.1f}°")
                if any(abs(angle) > 90 for angle in [result['pitch'], result['yaw'], result['roll']]):
                    print(f"⚠️ {image_id} - Sanity Warning: Pose angles exceed ±90°")
                    print(f"{image_id} - Pitch: {result['pitch']:.1f}°, Yaw: {result['yaw']:.1f}°, Roll: {result['roll']:.1f}°")
                result_to_save = {
                    'encoding_id': int(encoding_id),
                    'image_id': int(image_id),
                    'pitch': result['pitch'],
                    'yaw': result['yaw'],
                    'roll': result['roll'],
                }
                batch_success.append(result_to_save)
                batch_stats['pose_saved'] += 1
            else:
                print(f"❌ {image_id} - Failed to calculate pose")
                batch_stats['skipped'] += 1
        # 4. Save batch_success to MySQL in one batch based on dictionary
        if images_hw_updates or batch_success:
            try:
                if images_hw_updates:
                    session.bulk_update_mappings(Images, images_hw_updates)
                if batch_success:
                    session.bulk_update_mappings(Encodings, batch_success)
                    # session.bulk_update_mappings(SegmentBig, batch_success)
                session.commit()
            except Exception as e:
                session.rollback()
                print(f"🚨 Batch commit failed: {type(e).__name__}: {e}")
        print(f"Saved {len(batch_success)} pose results to MySQL.")
        if images_hw_updates:
            print(f"Saved {len(images_hw_updates)} Images.h/w updates to MySQL.")

    print(
        f"Batch stats: queried={batch_stats['queried']}, "
        f"h_w_filled={batch_stats['h_w_filled']}, "
        f"h_w_fill_failed={batch_stats['h_w_fill_failed']}, "
        f"pose_saved={batch_stats['pose_saved']}, "
        f"skipped={batch_stats['skipped']}"
    )

    # session.commit()
    last_id = results[-1][1]
    print(f"Processed up to image_id = {last_id}")

    # for testing, break after one batch
    # break

session.close()
mongo_client.close()
engine.dispose()
# Note: Ensure that the MongoDB and MySQL connections are properly closed after processing.