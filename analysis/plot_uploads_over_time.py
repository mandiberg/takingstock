import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.pool import NullPool
from sqlalchemy import Column, Integer, String, Date, Boolean, ForeignKey

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

LIMIT = 140000000

# # Define the Images table
# class Images(Base):
#     __tablename__ = 'Images'

#     image_id = Column(Integer, primary_key=True)
#     uploadDate = Column(Date)

# Query the database for upload dates
results = session.query(Images.uploadDate).limit(LIMIT).all()
# .all()
df = pd.DataFrame(results, columns=['uploadDate'])

# Ensure uploadDate is sorted
if not df.empty:
    df['uploadDate'] = pd.to_datetime(df['uploadDate'], errors='coerce')
    df = df.dropna().sort_values('uploadDate')
    df = df[(df['uploadDate'] >= '1990-01-01') & (df['uploadDate'] <= '2024-12-31')]
    df['year'] = df['uploadDate'].dt.year

    # Group by date and count the number of uploads per day
    # upload_counts = df.groupby('uploadDate').size()
    upload_counts = df.groupby('year').size()

    # Plot the data
    plt.figure(figsize=(12, 6))
    plt.plot(upload_counts.index, upload_counts.values, marker='o', linestyle='-', color='blue')

    # Add labels and title
    plt.title('Uploads Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Uploads')
    plt.grid(True, linestyle='--', alpha=0.7)

    # Show the plot
    plt.tight_layout()
    plt.show()
else:
    print("No data available to plot.")

# Close the session
session.close()
engine.dispose()
