import os
import sqlalchemy
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker,scoped_session,declarative_base
#from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.pool import NullPool
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# importing from another folder
import sys

Repo_path='/Users/jhash/Documents/GitHub/facemap2/'
#Repo_path='/Users/michaelmandiberg/Documents/GitHub/facemap/'
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1,Repo_path )

from mp_db_io import DataIO
from my_declarative_base import Images, Encodings, Base, Column, Integer, String, Date, Boolean, DECIMAL, BLOB, ForeignKey, JSON, Float
from my_declarative_base import ImagesBG  # Replace 'your_module' with the actual module where your SQLAlchemy models are defined


######## Michael's Credentials ########
# platform specific credentials
# io = DataIO()
# db = io.db
# # overriding DB for testing
# io.db["name"] = "stock"
# ROOT = io.ROOT 
# NUMBER_OF_PROCESSES = io.NUMBER_OF_PROCESSES

#######################################

######## Satyam's Credentials ########
# platform specific credentials
IS_SSD = True
io = DataIO(IS_SSD)
db = io.db
# overriding DB for testing
io.db["name"] = "ministock"
ROOT = io.ROOT 
NUMBER_OF_PROCESSES = io.NUMBER_OF_PROCESSES
#######################################

LIMIT= 20000

engine = create_engine("mysql+pymysql://{user}:{pw}@{host}/{db}".format(host=db['host'], db=db['name'], user=db['user'], pw=db['pass']), poolclass=NullPool)

session = scoped_session(sessionmaker(bind=engine))
num_threads = 1


# Define the function for generating imagename
def create_BG_df():
    # Define the select statement to fetch all columns from the table
    images_bg = ImagesBG.__table__

    # Construct the select query
    #query = select([images_bg]) ## this DOESNT work on windows somehow
    query = select(images_bg).filter(ImagesBG.hue != None)

    # Optionally limit the number of rows fetched
    if LIMIT:
        query = query.limit(LIMIT)

    # Execute the query and fetch all results
    result = session.execute(query).fetchall()

    results=[]
    counter = 0

    for row in result:
        image_id =row[0]
        if row[3] > 0:
            hue = row[3]
            lum = row[4]
        else:
            hue = row[1]
            lum = row[2]
        #print(hue,lum)
        results.append({"image_id": image_id, "hue": hue, "luminosity": lum})
    
    df = pd.DataFrame(results)
    return df

def save_hist(df,column):
    folder_path = os.path.join(os.getcwd(), 'plots')
    file_path= os .path.join(folder_path,'hist_'+column+'.png')

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

def save_scatter(df,column1,column2):
    folder_path = os.path.join(os.getcwd(), 'plots')
    file_path= os.path.join(folder_path,'Scatter_'+column1+column2+'.png')

    # Plot histogram
    plt.scatter(df[column1],df[column2])
    plt.xlabel(column1)
    plt.ylabel(column2)
    plt.title('Scatter plot of '+column1+column2)
    plt.grid(True)

    # Save the histogram as a PNG file
    plt.savefig(file_path)
    plt.close()

    return


try:
    df=create_BG_df()
    print(f"dataframe created.")
    print(df)
    save_hist(df,"luminosity")
    print("histogram saved")
    save_scatter(df,"luminosity","hue")
    print("scatter plot created")
    

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # Close the session
    session.close()