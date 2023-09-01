import folium
import numpy as np
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker, aliased
from my_declarative_base import Images, Encodings  # Adjust the import as needed
from sqlalchemy.pool import NullPool
from mp_db_io import DataIO

# Setup your credentials and database connection as before
io = DataIO()
db = io.db
engine = create_engine("mysql+pymysql://{user}:{pw}@/{db}?unix_socket={socket}".format(
    user=db['user'], pw=db['pass'], db=db['name'], socket=db['unix_socket']
), poolclass=NullPool)
Session = sessionmaker(bind=engine)
session = Session()

# Define the query to retrieve face_x and face_y from Encodings
images_alias = aliased(Images)
encodings_alias = aliased(Encodings)
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

# Set up the Folium map centered around the data
m = folium.Map(location=[np.mean(face_y), np.mean(face_x)], zoom_start=10)

# Create a folium HeatMap layer with the heatmap data
heat_data = [[[point[0], point[1]] for point in zip(yedges, row)] for row in heatmap]
folium.plugins.HeatMap(heat_data).add_to(m)

# Display the map
m.save("heatmap_map.html")
