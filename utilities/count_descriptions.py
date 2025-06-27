import sqlalchemy
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool
import pandas as pd
import os

# importing project-specific models
import sys
sys.path.insert(1, '/Users/michaelmandiberg/Documents/GitHub/facemap/')
from my_declarative_base import SegmentTable, Encodings, Base

# MySQL setup (preserving credentials framework)
from mp_db_io import DataIO
io = DataIO()
db = io.db
engine = create_engine(
    f"mysql+pymysql://{db['user']}:{db['pass']}@/{db['name']}?unix_socket={db['unix_socket']}",
    poolclass=NullPool
)
Session = sessionmaker(bind=engine)
session = Session()

COUNT_KEYS = True # Defaults to counting descriptions

if COUNT_KEYS:
    folder = "keywords"
    sql = text("""
        SELECT k.keyword_text, COUNT(*) AS count
        FROM ImagesTopics it
        JOIN Images i ON it.image_id = i.image_id
        JOIN ImagesKeywords ik ON it.image_id = ik.image_id
        JOIN Keywords k ON ik.keyword_id = k.keyword_id
        WHERE it.topic_id = :topic_id
        GROUP BY k.keyword_text
        ORDER BY count DESC
    """)
else:
    folder = "descriptions"
    sql = text("""
        SELECT i.description AS description, COUNT(*) AS count
        FROM ImagesTopics it
        JOIN Images i ON it.image_id = i.image_id
        WHERE it.topic_id = :topic_id
        GROUP BY i.description
        ORDER BY count DESC
    """)

    # Set where you want to save the output
ROOT_FOLDER_PATH = '/Users/michaelmandiberg/Documents/projects-active/facemap_production/description_counts_bytopic'
save_folder_path = os.path.join(ROOT_FOLDER_PATH, folder)
os.makedirs(save_folder_path, exist_ok=True)

# Loop through all topic_ids
for topic_id in range(64):
# for topic_id in [57,58]:
    print(f"Processing topic_id {topic_id}...")
    
    
    # Run query and fetch into pandas DataFrame
    result = session.execute(sql, {'topic_id': topic_id})
    df = pd.DataFrame(result.fetchall(), columns=['description', 'count'])

    # Reorder columns to match request
    df = df[['count', 'description']]

    # Save to CSV
    output_path = os.path.join(save_folder_path, f"topic_{topic_id:02d}_counts.csv")
    df.to_csv(output_path, index=False)

session.close()
engine.dispose()
