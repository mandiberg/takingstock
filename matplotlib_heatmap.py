import matplotlib.pyplot as plt
import numpy as np
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from my_declarative_base import Images, Encodings  # Adjust the import as needed

#mine
from mp_db_io import DataIO

######## Michael's Credentials ########
# platform specific credentials
io = DataIO()
db = io.db
# overriding DB for testing
# io.db["name"] = "gettytest3"
ROOT = io.ROOT 
NUMBER_OF_PROCESSES = io.NUMBER_OF_PROCESSES
#######################################


engine = create_engine("mysql+pymysql://{user}:{pw}@/{db}?unix_socket={socket}".format(
    user=db['user'], pw=db['pass'], db=db['name'], socket=db['unix_socket']
), poolclass=NullPool)

# metadata = MetaData(engine)
Session = sessionmaker(bind=engine)
session = Session()
# Base = declarative_base()


# Define the query to retrieve face_x and face_y from Images and Encodings
query = select(Images.face_x, Images.face_y).join(Encodings, Images.image_id == Encodings.image_id).\
        where(Images.site_name_id == 8, ~Images.age_id.in_([1, 2, 3, 4])).\
        limit(1000)

        # where(Images.site_name_id == 8, ~Images.age_id.in_([1, 2, 3, 4])).\


result = session.execute(query)
data = result.fetchall()


# Extract face_x and face_y from the retrieved data
face_x = [row[0] for row in data]
face_y = [row[1] for row in data]

# Create a heatmap using numpy's histogram2d
heatmap, xedges, yedges = np.histogram2d(face_x, face_y, bins=100)

# Transpose the heatmap for proper orientation
heatmap = heatmap.T

# Set up the figure and axis for plotting
plt.figure(figsize=(10, 8))
plt.title('Face Location Heatmap')
plt.xlabel('Face X')
plt.ylabel('Face Y')

# Create the heatmap plot
plt.imshow(heatmap, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin='lower', cmap='inferno')

# Add a colorbar
cbar = plt.colorbar()
cbar.set_label('Frequency')

# Display the plot
plt.show()
