import matplotlib.pyplot as plt
import numpy as np
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from my_declarative_base import Images, Encodings  # Adjust the import as needed
from sqlalchemy.pool import NullPool
from sqlalchemy.orm import aliased



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


# Create aliases for the tables to differentiate between Images and Encodings
images_alias = aliased(Images)
encodings_alias = aliased(Encodings)

# Define the query to retrieve face_x and face_y from Encodings
query = select(encodings_alias.face_x, encodings_alias.face_y).\
        join(images_alias, images_alias.image_id == encodings_alias.image_id).\
        where(images_alias.site_name_id == 3, ~images_alias.age_id.in_([1, 2, 3, 4]), encodings_alias.is_face.is_(True)).\
        limit(100000000)

result = session.execute(query)
data = result.fetchall()

# Convert Decimal values to float
face_x = [float(row[0]) for row in data]
face_y = [float(row[1]) for row in data]

# Create a heatmap using numpy's histogram2d
heatmap, xedges, yedges = np.histogram2d(face_y, face_x, bins=500)

# # Filter out low-frequency points
# min_frequency = 100
# heatmap_filtered = np.where(heatmap >= min_frequency, heatmap, np.nan)

# Transpose the heatmap for proper orientation
heatmap = heatmap.T

# Set up the figure and axis for plotting
plt.figure(figsize=(10, 8))
plt.title('Face Location Heatmap')
plt.xlabel('Face Y - Left to Right')
plt.ylabel('Face X - Up and Down')

# Create the heatmap plot
plt.imshow(heatmap, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin='lower', cmap='inferno')

# Add a colorbar
cbar = plt.colorbar()
cbar.set_label('Frequency')

# Display the plot
plt.show()
