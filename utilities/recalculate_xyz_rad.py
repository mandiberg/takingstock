import os
import pickle
import cv2
import numpy as np
import sqlalchemy
from sqlalchemy import Float, create_engine, Column, Integer, Boolean, String
from sqlalchemy.orm import sessionmaker, aliased
from sqlalchemy.pool import NullPool

# importing project-specific models
import sys
sys.path.insert(1, '/Users/michaelmandiberg/Documents/GitHub/facemap/')
from my_declarative_base import SegmentBig, SegmentTable, Encodings, Base, Images
from mp_pose_est import SelectPose

# MongoDB setup
import pymongo
mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
mongo_db = mongo_client["stock"]
mongo_collection = mongo_db["encodings"]  # adjust collection name if needed

# MySQL setup (preserving credentials framework)
from mp_db_io import DataIO
io = DataIO()
db = io.db
engine = create_engine(
    f"mysql+pymysql://{db['user']}:{db['pass']}@/{db['name']}?unix_socket={db['unix_socket']}",
    poolclass=NullPool
)
Session = sessionmaker(bind=engine)
session = Session()

# Batch processing parameters
batch_size = 1000
last_id = 112245607 # this is now image_id, not encoding_id
VERBOSE = False

HelperTable_name = "SegmentHelper_sept2025_heft_keywords"
class HelperTable(Base):
    __tablename__ = HelperTable_name
    seg_image_id=Column(Integer,primary_key=True, autoincrement=True)
    image_id = Column(Integer, primary_key=True, autoincrement=True)
class ImagesArmsPoses3D(Base):
    __tablename__ = "ImagesArmsPoses3D"
    image_id = Column(Integer, primary_key=True)
    cluster_id = Column(Integer)
    cluster_dist = Column(Float)

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
    results = (
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
        )
        # .join(s, s.image_id == Encodings.image_id)
        # .join(ihp, s.image_id == ihp.image_id)
        .join(Images, Images.image_id == Encodings.image_id)
        .filter(
            Encodings.mongo_face_landmarks.is_(True),
            Encodings.image_id > last_id,
            Encodings.pitch.is_(None),
            Images.h.isnot(None),
            Images.w.isnot(None),
        )
        .limit(batch_size)
        .all()
    )
    # .order_by(Encodings.image_id)
        # Encodings.is_feet.is_(None),

    if not results:
        print("No more rows to process. Exiting.")
        break
    
    print(f"Processing {len(results)} records...")

    # Process in batches
    for i in range(0, len(results), batch_size):
        batch = results[i:i + batch_size]
        batch_success = []
        for encoding_id, image_id, face_x, face_y, face_z, bbox, is_feet, h, w in batch:
            # 2. Fetch pickled face_landmarks from Mongo
            mongo_doc = mongo_collection.find_one({"image_id": image_id}, {"face_landmarks": 1})
            is_feet = False
            faceLms = None
            if mongo_doc and mongo_doc.get("face_landmarks"):
                # 3. Unpickle to MediaPipe object
                faceLms = pickle.loads(mongo_doc["face_landmarks"])
            else:
                print(f"No face_landmarks found in MongoDB for image_id {image_id}. Skipping.")
                continue

            # create a blank (all black) image of dimensions h x w
            # validate h and w (they may be NULL in the DB)
            if not (isinstance(h, int) and isinstance(w, int) and h > 0 and w > 0):
                print(f"Skipping image_id {image_id}: invalid image dimensions h={h}, w={w}")
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
            else:
                print(f"❌ {image_id} - Failed to calculate pose")
        # 4. Save batch_success to MySQL in one batch based on dictionary
        if batch_success:
            # print(f"Saving {len(batch_success)} pose results to MySQL...")
            # print(f"these are the image_ids: {[item['image_id'] for item in batch_success]}")
            session.bulk_update_mappings(Encodings, batch_success)
            # session.bulk_update_mappings(SegmentBig, batch_success)
            session.commit()
        print(f"Saved {len(batch_success)} pose results to MySQL.")

    # session.commit()
    last_id = results[-1][1]
    print(f"Processed up to image_id = {last_id}")

    # for testing, break after one batch
    # break

session.close()
mongo_client.close()
engine.dispose()
# Note: Ensure that the MongoDB and MySQL connections are properly closed after processing.