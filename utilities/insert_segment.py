import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.pool import NullPool

# go get IO class from parent folder
# caution: path[0] is reserved for script path (or '' in REPL)
import sys
sys.path.insert(1, '/Users/michaelmandiberg/Documents/GitHub/facemap/')
# import file
from my_declarative_base import Base, Encodings, Images, Column, Integer, DECIMAL, BLOB, String, JSON
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


# THIS MAY BE DEPRECATED, AS I INSERT INTO SEGMENTS VIA SQL IN WORKBENCH

SegmentTable_name = 'SegmentAug30Straightahead'  #actually straight ahead smile


# Connect to the database
engine = create_engine("mysql+pymysql://{user}:{pw}@/{db}?unix_socket={socket}".format(
    user=db['user'], pw=db['pass'], db=db['name'], socket=db['unix_socket']
), poolclass=NullPool)

# metadata = MetaData(engine)
Session = sessionmaker(bind=engine)
session = Session()
Base = declarative_base()

# to create new SegmentTable with variable as name
class SegmentTable(Base):
    __tablename__ = SegmentTable_name

    image_id = Column(Integer, primary_key=True)
    site_name_id = Column(Integer)
    contentUrl = Column(String(300))
    imagename = Column(String(200))
    site_image_id = Column(String(50))
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

xmin = -33
xmax = -29
ymin = -15
ymax = 15
step = 2

# Define the batch size
batch_size = 1000  # Adjust this based on your memory and performance needs

# Iterate through each angle
for angle in range(xmin, xmax, step):
    print("querying for ", str(angle))
    # Set the filters for Encodings
    encodings_filters = [
        Encodings.face_encodings68.isnot(None),
        Encodings.face_x > angle, Encodings.face_x <= angle+step,
        Encodings.face_y > ymin, Encodings.face_y < ymax,
        Encodings.face_z > -2, Encodings.face_z < 2
    ]

    results = session.query(
        Encodings.image_id, Encodings.face_x, Encodings.face_y,
        Encodings.face_z, Encodings.mouth_gap, Encodings.face_landmarks,
        Encodings.bbox, Encodings.face_encodings68
    ).filter(*encodings_filters).all()
    print("returned this many results ", str(len(results)))

    batch = []  # Accumulate entries for batch processing

    # Iterate through results and accumulate entries in the batch
    for result in results:
        image_id = result[0]  # Extract image_id from the result tuple
        image_data = session.query(Images).filter(Images.image_id == image_id).first()

        if image_data:
            segment_entry = SegmentTable(
                image_id=image_id,
                site_name_id=image_data.site_name_id,
                contentUrl=image_data.contentUrl,
                imagename=image_data.imagename,
                site_image_id=image_data.site_image_id,
                age_id = image_data.age_id,
                age_detail_id = image_data.age_detail_id,
                gender_id = image_data.gender_id,
                location_id = image_data.location_id,
                face_x=result[1],
                face_y=result[2],
                face_z=result[3],
                mouth_gap=result[4],
                face_landmarks=result[5],
                bbox=result[6],
                face_encodings68=result[7]
            )
            batch.append(segment_entry)

            # Commit the batch when it reaches the desired size
            if len(batch) >= batch_size:
                session.add_all(batch)
                session.commit()
                print("Committed batch for angle", angle)
                batch = []  # Clear the batch

    # Commit any remaining entries in the batch
    if batch:
        session.add_all(batch)
        session.commit()
        print("Committed remaining batch for angle", angle)
