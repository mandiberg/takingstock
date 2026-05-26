# Query: images per canonical signature cluster_id for a given topic

from sqlalchemy import create_engine, text,select, MetaData, Table, Column, Numeric, Integer, VARCHAR, update, Float
from sqlalchemy.pool import NullPool
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
# my ORM
from my_declarative_base import Base, Encodings, ImagesTopics, SegmentTable,ImagesBackground, Hands, Column, Integer, String, Date, Boolean, DECIMAL, BLOB, ForeignKey, JSON, Images, Detections


from sqlalchemy import text
from mp_db_io import DataIO
from tools_clustering import ToolsClustering

IS_SSD = False
VERBOSE = True
SSD_PATH = "/mnt/ssd1/"
# I/O utils
io = DataIO(IS_SSD, VERBOSE=VERBOSE, SSD_PATH=SSD_PATH)
db = io.db

CLUSTER_TYPE = "ArmsPoses3D_ObjectFusion"  # TEST: new Arms/ObjectFusion mode
SORT_TYPE = "object_fusion" # for ArmsPoses3D_ObjectFusion keep SORT_TYPE as object_fusion
SORT_TYPE_NONEOBJECT = "object_fusion" # sort used when pose_no is in OBJECT_NONE_CLUSTERS
USE_HSV = False

cl = ToolsClustering(SORT_TYPE, VERBOSE=VERBOSE)

if db['unix_socket']:
    # for MM's MAMP config
    engine = create_engine("mysql+pymysql://{user}:{pw}@/{db}?unix_socket={socket}".format(
        user=db['user'], pw=db['pass'], db=db['name'], socket=db['unix_socket']
    ), poolclass=NullPool)
else:
    engine = create_engine("mysql+pymysql://{user}:{pw}@{host}/{db}"
                                .format(host=db['host'], db=db['name'], user=db['user'], pw=db['pass']), poolclass=NullPool)
# metadata = MetaData(engine)
Session = sessionmaker(bind=engine)
session = Session()
Base = declarative_base()

rows = session.execute(text("""
    SELECT ios.cluster_id, COUNT(*) AS cnt
    FROM ImagesObjectSignatures ios
    JOIN ImagesTopics it ON ios.image_id = it.image_id
    WHERE it.topic_id = :topic_id
    GROUP BY ios.cluster_id
"""), {'topic_id': 11}).fetchall()

topic_sig_counts = {int(r.cluster_id): int(r.cnt) for r in rows}

# Then compute the collapse mapping
result = cl.compute_topic_collapse_mapping(topic_sig_counts, collapse_min=400)

# result['mapping']   → {under-supported cluster_id: target cluster_id}
# result['retained']  → cluster_ids already above COLLAPSE_MIN (use as-is)
# result['discarded'] → cluster_ids too small with no valid collapse target
# result['stats']     → summary counts

print("Collapse mapping:", result['mapping'])
print("Retained clusters:", result['retained'])
print("Discarded clusters:", result['discarded'])
print("Stats:", result['stats'])