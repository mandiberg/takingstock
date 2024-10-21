import pandas as pd
import matplotlib.pyplot as plt
import os

import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.pool import NullPool


import sys
if sys.platform == "darwin": sys.path.insert(1, '/Users/michaelmandiberg/Documents/GitHub/facemap/')
elif sys.platform == "win32": sys.path.insert(1, 'C:/Users/jhash/Documents/GitHub/facemap2/')
from mp_db_io import DataIO
from my_declarative_base import Images, Base, SegmentTable, Encodings, Clusters, Column, Integer, String, Date, Boolean, DECIMAL, BLOB, ForeignKey, JSON, Float

io = DataIO()
db = io.db

io.db["name"] = "stock"
ROOT = io.ROOT 
NUMBER_OF_PROCESSES = io.NUMBER_OF_PROCESSES
#######################################


engine = create_engine("mysql+pymysql://{user}:{pw}@/{db}?unix_socket={socket}".format(
    user=db['user'], pw=db['pass'], db=db['name'], socket=db['unix_socket']
), poolclass=NullPool)

# metadata = MetaData(engine)
Session = sessionmaker(bind=engine)
session = Session()
Base = declarative_base()

IS_CSV = False

# CLUSTER_TYPE = "Clusters"
# CLUSTER_TYPE = "BodyPoses"
# CLUSTER_TYPE = "HandsPositions"
CLUSTER_TYPE = "HandsGestures"
ClustersTable_name = CLUSTER_TYPE
ImagesClustersTable_name = "Images"+CLUSTER_TYPE

class ImagesClusters(Base):
    __tablename__ = ImagesClustersTable_name

    image_id = Column(Integer, ForeignKey(Images.image_id, ondelete="CASCADE"), primary_key=True)
    cluster_id = Column(Integer, ForeignKey(f'{ClustersTable_name}.cluster_id', ondelete="CASCADE"))
    cluster_dist = Column(Float)


if IS_CSV:
    METHOD="meta" ##openai or bark or meta
    INPUT = os.path.join(io.ROOT, "audioproduction", METHOD, "metas.csv")
    # Read the CSV file
    df = pd.read_csv(INPUT)
    plot_column = 'topic_fit'

else:

    results = session.query(ImagesClusters.cluster_dist).filter(ImagesClusters.cluster_id == 13).all()
    # results = session.query(ImagesClusters.cluster_dist).all()
    df = pd.DataFrame(results)
    plot_column = 'cluster_dist'
print(df.columns)

# Create the histogram
plt.figure(figsize=(10, 6))
plt.hist(df[plot_column], bins=20, edgecolor='black')

# Add labels and title
plt.xlabel(plot_column)
plt.ylabel('Frequency')
title = 'Distribution of Scores'+plot_column
plt.title(title)

# Add grid lines
plt.grid(True, linestyle='--', alpha=0.7)

# Show the plot
plt.show()

session.close()
engine.dispose()