import os
import sqlalchemy
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker,scoped_session,declarative_base
#from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.pool import NullPool
import pandas as pd
import matplotlib.pyplot as plt
import colorsys

import numpy as np
# importing from another folder
import sys
from matplotlib.patches import Rectangle
import pickle
import json

# Repo_path='/Users/jhash/Documents/GitHub/facemap2/'
Repo_path='/Users/michaelmandiberg/Documents/GitHub/facemap/'
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1,Repo_path )

from mp_db_io import DataIO
from my_declarative_base import Images, Encodings, Base, Column, Integer, String, Date, Boolean, DECIMAL, BLOB, ForeignKey, JSON, Float
from my_declarative_base import ImagesBackground  # Replace 'your_module' with the actual module where your SQLAlchemy models are defined


######## Michael's Credentials ########
# platform specific credentials
IS_SSD = True
io = DataIO(IS_SSD)
db = io.db
# overriding DB for testing
io.db["name"] = "stock"
ROOT = io.ROOT 
NUMBER_OF_PROCESSES = io.NUMBER_OF_PROCESSES

#######################################

# ######## Satyam's Credentials ########
# # platform specific credentials
# IS_SSD = True
# io = DataIO(IS_SSD)
# db = io.db
# # overriding DB for testing
# io.db["name"] = "ministock"
# ROOT = io.ROOT 
# NUMBER_OF_PROCESSES = io.NUMBER_OF_PROCESSES
# #######################################

LIMIT= 250000

engine = create_engine("mysql+pymysql://{user}:{pw}@{host}/{db}".format(host=db['host'], db=db['name'], user=db['user'], pw=db['pass']), poolclass=NullPool)

session = scoped_session(sessionmaker(bind=engine))
num_threads = 1
folder_path = os.path.join(os.getcwd(), 'analysis/plots')
Base = declarative_base()

class SegmentOct20(Base):
    __tablename__ = 'SegmentOct20'
    seg_image_id=Column(Integer,primary_key=True, autoincrement=True)
    image_id = Column(Integer)
    site_name_id = Column(Integer)
    site_image_id = Column(String(50),nullable=False)
    contentUrl = Column(String(300), nullable=False)
    imagename = Column(String(200))
    age_id = Column(Integer)
    age_detail_id = Column(Integer)
    gender_id = Column(Integer)
    location_id = Column(Integer)
    face_x = Column(DECIMAL(6, 3))
    face_y = Column(DECIMAL(6, 3))
    face_z = Column(DECIMAL(6, 3))
    mouth_gap = Column(DECIMAL(6, 3))
    face_landmarks = Column(BLOB)
    bbox = Column(JSON)
    face_encodings = Column(BLOB)
    face_encodings68 = Column(BLOB)
    body_landmarks = Column(BLOB)



# Define the function for generating imagename
def create_BG_df():
    # Define the select statement to fetch all columns from the table
    images_bg = ImagesBackground.__table__

    # Construct the select query
    #query = select([images_bg]) ## this DOESNT work on windows somehow
    query = select(images_bg).filter(ImagesBackground.hue != None)

    # Optionally limit the number of rows fetched
    if LIMIT:
        query = query.limit(LIMIT)

    # Execute the query and fetch all results
    result = session.execute(query).fetchall()

    results=[]
    counter = 0

    for row in result:
        image_id =row[0]
        if row[4] >0:
            hue = row[4]
            lum = row[5]
            sat = row[6]
        else:
            hue = row[1]
            lum = row[2]
            sat = row[3]
        print(hue,lum,sat)
        results.append({"image_id": image_id, "hue": hue, "luminosity": lum,"sat":sat})
    
    df = pd.DataFrame(results)
    return df

def create_segment_df():
    # Define the select statement to fetch all columns from the table
    segment = SegmentOct20.__table__

    # Construct the select query
    #query = select([images_bg]) ## this DOESNT work on windows somehow
    query = select(segment)

    # Optionally limit the number of rows fetched
    if LIMIT:
        query = query.limit(LIMIT)

    # Execute the query and fetch all results
    result = session.execute(query).fetchall()

    results=[]
    counter = 0

    for counter,row in enumerate(result):
        if counter%1000==0:print(counter,"rows made")
        image_id =row[0]
        sitename_id=row[2]
        face_x,face_y,face_z=row[10],row[11],row[12]
        mouth_gap=row[13]
        bbox=row[15]
        if type(bbox)==str:
            bbox=json.loads(bbox)
             
        #print(type(bbox),bbox)
        face_encodings=pickle.loads(row[17])
        results.append({"image_id": image_id, "sitename_id": sitename_id, "face_x": face_x,"face_y":face_y,"face_z":face_z,"mouth_gap":mouth_gap,"bbox":bbox,"face_encodings":face_encodings})
    
    df = pd.DataFrame(results)
    return df

def plot_bbox(df):
    file_path= os .path.join(folder_path,'bbox.png')

    fig, ax = plt.subplots()
    ax.plot([0, 0],[0, 0],color="white")

    for bbox in df['bbox']:
        width,height=bbox['right']-bbox['left'],bbox['top']-bbox['bottom']
        ax.add_patch(Rectangle((bbox['left'],bbox['bottom'] ), width, height,alpha=0.01))
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('bbox plot')
 
    plt.savefig(file_path)
    plt.close()
    return

def save_hist(df,column):
    file_path= os .path.join(folder_path,'hist_'+column+'.png')
    if column=="sat":df[column]*=1000
    # Plot histogram
    bin_edges=np.arange(min(df[column])//1,max(df[column])//1 + 1)
    plt.hist(df[column], bins=bin_edges, edgecolor='black')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.title('Histogram of'+column)
    plt.grid(True)

    # Save the histogram as a PNG file
    plt.savefig(file_path)
    plt.close()
    return 

def save_scatter(df,column1,column2,column3):
    folder_path = os.path.join(os.getcwd(), 'analysis/plots')
    file_path= os.path.join(folder_path,'Scatter_'+column1+column2+'.png')

    # print the columsn in the df
    print(df.columns)


    def rgb_to_hex(rgb):
        """Convert RGB tuple to hex color code."""
        return '#{:02x}{:02x}{:02x}'.format(*rgb)

    def hls_to_hex(hls):
        """Convert HLS tuple to hex color code."""
        rgb = colorsys.hls_to_rgb(*hls)
        return rgb_to_hex(tuple(int(255 * x) for x in rgb))

    # Convert hue, luminosity, and saturation to hex color code
    df['hex_color'] = df.apply(lambda row: hls_to_hex((row['hue'] / 360, row['luminosity'] / 100, row['sat'])), axis=1)

    # Display the DataFrame with the new hex color column
    print(df)
    

    # Plot scatter with combined color
    plt.scatter(df[column1], df[column2], c=df['hex_color'], alpha=0.005, cmap='hsv')
    # Plot histogram
    # plt.scatter(df[column1],df[column2],c=df[column3],alpha=0.005,cmap='hsv')
    plt.xlabel(column1)
    plt.ylabel(column2)
    plt.title('Scatter plot of '+column1+column2 +"with color from"+column3+"with dtapoints" +str(len(df[column1])))
    plt.grid(True)

    # Save the histogram as a PNG file
    plt.savefig(file_path)
    plt.close()

    return


df=create_BG_df()
# df=create_segment_df()
print(f"dataframe created.")
print(df)
# save_hist(df,"luminosity")
# save_hist(df,"hue")
# save_hist(df,"sat")
# print("histogram saved")
# save_scatter(df,"sat","hue","hue")
save_scatter(df,"luminosity","sat","hue")
# print("scatter plot created")
# plot_bbox(df)
# print("bbox plot created")
   
#print(f"An error occurred: {e}")

#finally:
    # Close the session
session.close()