import pickle
import sqlalchemy
# from sqlalchemy import create_engine
# from sqlalchemy.orm import sessionmaker
# from my_declarative_base import Encodings

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
# my ORM
from my_declarative_base import Base, Encodings, Column, Integer, String, Date, Boolean, DECIMAL, BLOB, ForeignKey, JSON

from mp_db_io import DataIO

# script that will open Protocol=3 pickles
# repickle them as Protocol=4, and store them back in the database. 

# I/O utils
io = DataIO(IS_SSD)
db = io.db
NUMBER_OF_PROCESSES = io.NUMBER_OF_PROCESSES


def repickle_encodings():
    # Create a SQLAlchemy engine and session
    engine = create_engine("mysql+pymysql://{user}:{pw}@{host}/{db}"
                                    .format(host=db['host'], db=db['name'], user=db['user'], pw=db['pass']), poolclass=NullPool)
    # metadata = MetaData(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    Base = declarative_base()

    # Fetch all records from the Encodings table
    encodings = session.query(Encodings).all()

    for encoding in encodings:
        # Unpickle the face_landmarks, bbox, and face_encodings
        face_landmarks = pickle.loads(encoding.face_landmarks) if encoding.face_landmarks else None
        bbox = pickle.loads(encoding.bbox) if encoding.bbox else None
        face_encodings = pickle.loads(encoding.face_encodings) if encoding.face_encodings else None

        # Repickle the data with Protocol 4
        repickled_face_landmarks = pickle.dumps(face_landmarks, protocol=4) if face_landmarks else None
        repickled_bbox = pickle.dumps(bbox, protocol=4) if bbox else None
        repickled_face_encodings = pickle.dumps(face_encodings, protocol=4) if face_encodings else None

        # Update the corresponding rows in the database with the repickled data
        encoding.face_landmarks = repickled_face_landmarks
        encoding.bbox = repickled_bbox
        encoding.face_encodings = repickled_face_encodings

    # Commit the changes to the database
    session.commit()
    session.close()

# Run the repickling process
repickle_encodings()
