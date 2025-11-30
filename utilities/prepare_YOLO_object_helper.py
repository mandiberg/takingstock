import os
import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

# importing project-specific models
import sys
sys.path.insert(1, '/Users/michaelmandiberg/Documents/GitHub/facemap/')
from my_declarative_base import SegmentBig, Images, ImagesKeywords, Encodings, Base, YoloClasses, Detections, SegmentHelperObject

'''
0. cycle through the yolo_classes_dict and for each class, 
grep the KEYWORD_FILE for that class name (case insensitive), 
and write the results to a new file named "keyword_segment_<class_name>.csv"

1. open each csv in INGEST_FOLDDER, and build a mysql query to 
SELECT
'''

KEYWORD_FOLDER = "/Users/michaelmandiberg/Documents/GitHub/facemap/utilities/keys/"
KEYWORD_FILE = "Keywords_202408151415.csv"
INGEST_FOLDER = "/Users/michaelmandiberg/Documents/projects-active/facemap_production/keyword_segment_SSD_process/ingest_this/"
MODE = 1


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
last_id = 0


def parse_keyword_file_by_class():
    for class_id, class_name in yolo_classes_dict["class"].items():
        output_file = f"{KEYWORD_FOLDER}keyword_segment_{class_id}.csv"
        command = f'grep -i "{class_name}" {KEYWORD_FOLDER}{KEYWORD_FILE} > {output_file}'
        os.system(command)
        print(f"Wrote keywords for class {class_name} to file: {output_file}") 

def load_yolo_classes_dict():
    yolo_classes_in_db = {}
    results = session.query(YoloClasses).all()
    for row in results:
        yolo_classes_in_db[row.class_id] = row.class_name
    # print("Loaded YoloClasses from DB:", yolo_classes_in_db)
    return yolo_classes_in_db

def save_results_to_SegmentHelperObject(results, class_id):
    for result in results:
        segment_helper = SegmentHelperObject(
            image_id=result.image_id,
            site_name_id=result.site_name_id,
            imagename=result.imagename,
            class_id=class_id
        )
        session.add(segment_helper)
    session.commit()
    # print(f"Saved {len(results)} entries to SegmentHelperObject for class_id {class_id}.")


if __name__ == "__main__":

    yolo_classes_dict = load_yolo_classes_dict()
    if MODE == 0:
        parse_keyword_file_by_class()
    elif MODE == 1:
        for class_id in yolo_classes_dict.keys():
            # class_id = 73
            class_counter = 0
            keywords = io.get_csv_aslist(f"{INGEST_FOLDER}keyword_segment_{class_id}.csv")
            if not keywords:
                print(f"No keywords found for class_id {class_id}, skipping.")
                continue
            # Get min and max encoding_id for batching
            # min_id = session.query(sqlalchemy.func.min(Encodings.encoding_id)).scalar() or 0
            min_id = last_id

            # Process in batches, while there are still keywords to process
            while True:
                results = (
                    session.query(Images.image_id, Images.site_name_id, Images.imagename)
                    .join(ImagesKeywords, Images.image_id == ImagesKeywords.image_id)
                    .join(SegmentHelperObject, Images.image_id == SegmentHelperObject.image_id, isouter=True)
                    .join(Encodings, Images.image_id == Encodings.image_id)
                    .where((Encodings.is_face == 1) | (Encodings.is_body == 1))
                    .where(ImagesKeywords.keyword_id.in_(keywords))
                    .where(SegmentHelperObject.image_id.is_(None))  # only images not already in SegmentHelperObject
                    .filter(Images.image_id > min_id)
                    .limit(batch_size)
                    .all()
                )
                # print("Results:", results)

                if results:
                    # print(f"Found image_id: {result.image_id}, site_name_id: {result.site_name_id}, imagename: {result.imagename}")
                    # Here you would add your processing logic, e.g., checking MongoDB, updating MySQL, etc.
                    # For now, we just print the results.
                    save_results_to_SegmentHelperObject(results, class_id)
                    min_id = results[-1].image_id + 1  # Update min_id for next batch
                    class_counter += len(results)
                    print(f"Saved batch starting from image_id {min_id}, with {len(results)} images, total for class_id {class_id}: {class_counter}.")
                    
                    # break  # temporary for testing
                else:
                    print("No more images found for the given keywords. Exiting.")
                    break  # temporary for testing


session.close()
