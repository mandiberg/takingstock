import sqlalchemy
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool
import pandas as pd
import os

# importing project-specific models
import sys

sys.path.insert(1, '/Users/michaelmandiberg/Documents/GitHub/takingstock/')
from my_declarative_base import SegmentTable, Encodings, Base
from constants_make_video import *

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

# COUNT_KEYS = True # Defaults to counting descriptions
PAIR_LIST = FUSION_PAIR_DICT_DETECTIONS_THEOFFICE[0] # set to None to process all topics, or set to a list of topic_ids to process specific topics

print(f"PAIR_LIST: {PAIR_LIST}")
# for each pair in pair list, query the database for the count of images that match armspose3D cluster and signature
# print out each count
for arms_cluster, signature in PAIR_LIST:
    print(f"Processing pair arms_cluster {arms_cluster} and signature {signature}")
    sql = text("""
        SELECT COUNT(*) as ccount
        FROM Images i
        JOIN ImagesArmsposes3D ia ON i.image_id = ia.image_id
        JOIN ImagesObjectSignatures ios ON i.image_id = ios.image_id
        WHERE ia.cluster_id = :arms_cluster
        AND ios.cluster_id = :signature
    """)
    result = session.execute(sql, {"arms_cluster": arms_cluster, "signature": signature}).fetchone()
    print(f"{arms_cluster}, {signature}: {result}")
    # count = result["ccount"]
    # print(f"Count for arms_cluster {arms_cluster} and signature {signature}: {count}")

session.close()
engine.dispose()
