from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool
from my_declarative_base import Images, BagOfKeywords,Keywords,ImagesKeywords,ImagesEthnicity  # Replace 'your_module' with the actual module where your SQLAlchemy models are defined
from mp_db_io import DataIO
import pickle
import numpy as np
from pick import pick

io = DataIO()
db = io.db
io.db["name"] = "ministock"

# Create a database engine
engine = create_engine("mysql+pymysql://{user}:{pw}@{host}/{db}".format(host=db['host'], db=db['name'], user=db['user'], pw=db['pass']), poolclass=NullPool)

# Create a session
Session = sessionmaker(bind=engine)
session = Session()

title = 'Please choose your operation: '
options = ['Create table', 'Fetch keywords list', 'Fetch ethnicity list']
option, index = pick(options, title)

LIMIT= 100000

## I've created this is 3 sections , and they work perfectly seperately, but i can't overwrite data if the table is already created
## I haven't been able to add this "feature" but im leaving it as it is now, i'll try to fix it later
if index == 0:
    ################# CREATE TABLE ###########
    # Define the columns you want to retrieve from Images table
    # columns = [Images.image_id, Images.description, Images.gender_id, Images.age_id, Images.location_id]

    # Build a select query for fetching data from Images table
    # select_query = select(Images.image_id, Images.description, Images.gender_id, Images.age_id, Images.location_id).select_from(Images).limit(LIMIT)
    # quit()

    # Build a select query for fetching data from Images table
    select_query = select(Images.image_id, Images.description, Images.gender_id, Images.age_id, Images.location_id).select_from(Images).outerjoin(BagOfKeywords, Images.image_id == BagOfKeywords.image_id).filter(BagOfKeywords.image_id == None).limit(LIMIT)

    # Fetch the data
    result = session.execute(select_query).fetchall()

    #Iterate through the fetched data and insert it into BagOfKeywords table
    for row in result:
        image_id, description, gender_id, age_id, location_id = row
        
        # Create a BagOfKeywords object
        bag_of_keywords = BagOfKeywords(
            image_id=image_id,
            description=description,
            gender_id=gender_id,
            age_id=age_id,
            location_id=location_id,
            keyword_list=None,  # Set this to None or your desired value
            ethnicity_list=None  # Set this to None or your desired value
        )
        print(f"Keyword list for image_id {image_id} updated successfully.")
        # Add the BagOfKeywords object to the session
        session.add(bag_of_keywords)

    # Commit the changes to the database
    session.commit()

if index == 1:
    ################FETCHING KEYWORDS####################################

    distinct_image_ids_query = select(BagOfKeywords.image_id.distinct()).filter(BagOfKeywords.keyword_list == None).limit(LIMIT)

    distinct_image_ids = [row[0] for row in session.execute(distinct_image_ids_query).fetchall()]

    # Iterate through each distinct image_id
    for target_image_id in distinct_image_ids:

        # Build a select query to retrieve keyword_ids for the specified image_id
        select_keyword_ids_query = (
            select(ImagesKeywords.keyword_id)
            .filter(ImagesKeywords.image_id == target_image_id)
        )

        # Execute the query and fetch the result as a list of keyword_ids
        result = session.execute(select_keyword_ids_query).fetchall()
        keyword_ids = [row.keyword_id for row in result]

        # Build a select query to retrieve keywords for the specified keyword_ids
        select_keywords_query = (
            select(Keywords.keyword_text)
            .filter(Keywords.keyword_id.in_(keyword_ids))
            .order_by(Keywords.keyword_id)
        )

        # Execute the query and fetch the results as a list of keyword_text
        result = session.execute(select_keywords_query).fetchall()
        keyword_list = [row.keyword_text for row in result]
        # Pickle the keyword_list
        keyword_list_pickle = pickle.dumps(keyword_list)

        # Update the BagOfKeywords entry with the corresponding image_id
        BOK_keywords_entry = (
            session.query(BagOfKeywords)
            .filter(BagOfKeywords.image_id == target_image_id)
            .first()
        )

        if BOK_keywords_entry:
            BOK_keywords_entry.keyword_list = keyword_list_pickle
            session.commit()
            print(f"Keyword list for image_id {target_image_id} updated successfully.")
        else:
            print(f"Keywords entry for image_id {target_image_id} not found.")
            
            
if index == 2:        
    distinct_image_ids_query = select(BagOfKeywords.image_id.distinct()).filter(BagOfKeywords.ethnicity_list == None).limit(LIMIT)

    distinct_image_ids = [row[0] for row in session.execute(distinct_image_ids_query).fetchall()]

    # Iterate through each distinct image_id
    for target_image_id in distinct_image_ids:

        #################FETCHING ETHNICITY####################################

        select_ethnicity_ids_query = (
            select(ImagesEthnicity.ethnicity_id)
            .filter(ImagesEthnicity.image_id == target_image_id)
        )

        result = session.execute(select_ethnicity_ids_query).fetchall()
        ethnicity_list = [row.ethnicity_id for row in result]

        ethnicity_list_pickle = pickle.dumps(ethnicity_list)

        # Update the BagOfKeywords entry with the corresponding image_id
        BOK_ethnicity_entry = (
            session.query(BagOfKeywords)
            .filter(BagOfKeywords.image_id == target_image_id)
            .first()
        )

        if BOK_ethnicity_entry:
            BOK_ethnicity_entry.ethnicity_list = ethnicity_list_pickle
            session.commit()
            print(f"Ethnicity list for image_id {target_image_id} updated successfully.")
        else:
            print(f"ethnicity entry for image_id {target_image_id} not found.")
    




print("done")
# Close the session
session.close()