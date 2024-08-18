import sqlalchemy
from sqlalchemy import create_engine, select, distinct, and_, or_, not_, update, insert, text
from sqlalchemy.orm import sessionmaker, aliased
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.pool import NullPool
from sqlalchemy.dialects.mysql import insert as mysql_insert

# go get IO class from parent folder
# caution: path[0] is reserved for script path (or '' in REPL)
import sys
sys.path.insert(1, '/Users/michaelmandiberg/Documents/GitHub/facemap/')
# import file
from my_declarative_base import Base, SegmentTable, ImagesKeywords, Encodings, Images, Column, Integer, DECIMAL, BLOB, String, JSON
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

#######################################
# AUG 2024 -- DO NOT USE THIS, USE 
# USE utilities/sql/make_segment_table.sql
# AT THE BOTTOM THERE ARE STEPS
#######################################

# ADD TO SEGMENTS WHEN SQL/WORKBENCH IS BORKING

HelperTable_name = "SegmentHelperMay24_allfingers"
MAKE_HELPERTABLE = True

xmin = -33
xmax = -29
ymin = -15
ymax = 15
zmin = -15
zmax = 15
step = 2
keyword_id = 908
kids_age_id = 3
LIMIT = 100


engine = create_engine("mysql+pymysql://{user}:{pw}@/{db}?unix_socket={socket}".format(
    user=db['user'], pw=db['pass'], db=db['name'], socket=db['unix_socket']
), poolclass=NullPool)

Base.metadata.bind = engine

Session = sessionmaker(bind=engine)
session = Session()

class HelperTable(Base):
    __tablename__ = HelperTable_name
    seg_image_id=Column(Integer,primary_key=True, autoincrement=True)
    image_id = Column(Integer, primary_key=True, autoincrement=True)

# Aliases for the tables
# Encodings = aliased(Encodings)
# SegmentTable = aliased(SegmentTable)
# ImagesKeywords = aliased(ImagesKeywords)
# Images = aliased(Images)
# HelperTable = aliased(HelperTable)

if MAKE_HELPERTABLE:
    # Define the subquery to match the SELECT DISTINCT part
    # subquery = select(Images.image_id).distinct() \
    #     .outerjoin(ImagesKeywords, Images.image_id == ImagesKeywords.image_id) \
    #     # .outerjoin(HelperTable, Images.image_id == HelperTable.image_id) \
    #     .where(
    #         ImagesKeywords.keyword_id == keyword_id,
    #         # HelperTable.image_id.is_(None)
    #     ).limit(LIMIT)

    # Define the subquery
    subquery = select(ImagesKeywords.image_id).distinct() \
        .where(
            ImagesKeywords.keyword_id == keyword_id,
        ).limit(LIMIT)

    # # Create a manual raw SQL query for INSERT IGNORE
    # insert_ignore_sql = text(f"""
    # INSERT IGNORE INTO {HelperTable.__tablename__} (image_id)
    # {str(subquery.compile(compile_kwargs={"literal_binds": True}))}
    # """)

    insert_stmt = insert(HelperTable).from_select(['image_id'], subquery).prefix_with('IGNORE')


    try:
        # Execute the raw SQL query
        session.execute(insert_stmt)
        session.commit()
    except Exception as e:
        session.rollback()
        print(f"An error occurred: {e}")
    finally:
        session.close()
    
else:
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
            Encodings.face_z > zmin, Encodings.face_z < zmax,
            Encodings.is_face.is_(True)
        ]

        images_filters = [
            SegmentTable.image_id.is_(None),
            ImagesKeywords.keyword_id == keyword_id,
            Images.age_id > kids_age_id,
        ]

        results = session.query(
            distinct(Images.image_id),
            Images.site_name_id,
            Images.site_image_id,
            Images.contentUrl,
            Images.imagename,
            Images.description,
            Encodings.face_x,
            Encodings.face_y,
            Encodings.face_z,
            Encodings.mouth_gap,
            Encodings.face_landmarks,
            Encodings.bbox,
            Encodings.face_encodings68,
            Encodings.body_landmarks
        ).outerjoin(Encodings, Images.image_id == Encodings.image_id) \
        .outerjoin(SegmentTable, Images.image_id == SegmentTable.image_id) \
        .outerjoin(ImagesKeywords, ImagesKeywords.image_id == Images.image_id) \
        .filter(and_(*encodings_filters), and_(*images_filters)) \
        .limit(LIMIT).all()

        # .all()
        # results = session.query(
        #     Encodings.image_id, Encodings.face_x, Encodings.face_y,
        #     Encodings.face_z, Encodings.mouth_gap, Encodings.face_landmarks,
        #     Encodings.bbox, Encodings.face_encodings68
        # ).filter(*encodings_filters)
        # .all()
        print("returned this many results ", str(len(results)))
        quit()
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
