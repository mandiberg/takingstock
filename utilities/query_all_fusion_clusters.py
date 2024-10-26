import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool
from sqlalchemy.ext.declarative import declarative_base

import sys
if sys.platform == "darwin": sys.path.insert(1, '/Users/michaelmandiberg/Documents/GitHub/facemap/')
elif sys.platform == "win32": sys.path.insert(1, 'C:/Users/jhash/Documents/GitHub/facemap2/')
from mp_db_io import DataIO
from my_declarative_base import Images, Base, SegmentTable, Encodings, Clusters, Column, Integer, String, Date, Boolean, DECIMAL, BLOB, ForeignKey, JSON, Float

io = DataIO()
db = io.db

io.db["name"] = "stock"
ROOT = io.ROOT 
NUMBER_OF_PROCESSES = io.NUMBER_OF_PROCESSES
#######################################

ROOT_FOLDER_PATH = '/Users/michaelmandiberg/Documents/projects-active/facemap_production/october_fusion_clusters'

# Create engine and session
engine = create_engine("mysql+pymysql://{user}:{pw}@/{db}?unix_socket={socket}".format(
    user=db['user'], pw=db['pass'], db=db['name'], socket=db['unix_socket']
), poolclass=NullPool)

Session = sessionmaker(bind=engine)
session = Session()

# SQL query template
sql_query_template = """
SELECT 
    ihp.cluster_id AS ihp_cluster,

    SUM(CASE WHEN ihg.cluster_id = 0 THEN 1 ELSE 0 END) AS ihg_0,
    SUM(CASE WHEN ihg.cluster_id = 1 THEN 1 ELSE 0 END) AS ihg_1,
    SUM(CASE WHEN ihg.cluster_id = 2 THEN 1 ELSE 0 END) AS ihg_2,
    SUM(CASE WHEN ihg.cluster_id = 3 THEN 1 ELSE 0 END) AS ihg_3,
    SUM(CASE WHEN ihg.cluster_id = 4 THEN 1 ELSE 0 END) AS ihg_4,
    SUM(CASE WHEN ihg.cluster_id = 5 THEN 1 ELSE 0 END) AS ihg_5,
    SUM(CASE WHEN ihg.cluster_id = 6 THEN 1 ELSE 0 END) AS ihg_6,
    SUM(CASE WHEN ihg.cluster_id = 7 THEN 1 ELSE 0 END) AS ihg_7,
    SUM(CASE WHEN ihg.cluster_id = 8 THEN 1 ELSE 0 END) AS ihg_8,
    SUM(CASE WHEN ihg.cluster_id = 9 THEN 1 ELSE 0 END) AS ihg_9,
    SUM(CASE WHEN ihg.cluster_id = 10 THEN 1 ELSE 0 END) AS ihg_10,
    SUM(CASE WHEN ihg.cluster_id = 11 THEN 1 ELSE 0 END) AS ihg_11,
    SUM(CASE WHEN ihg.cluster_id = 12 THEN 1 ELSE 0 END) AS ihg_12,
    SUM(CASE WHEN ihg.cluster_id = 13 THEN 1 ELSE 0 END) AS ihg_13,
    SUM(CASE WHEN ihg.cluster_id = 14 THEN 1 ELSE 0 END) AS ihg_14,
    SUM(CASE WHEN ihg.cluster_id = 15 THEN 1 ELSE 0 END) AS ihg_15,
    SUM(CASE WHEN ihg.cluster_id = 16 THEN 1 ELSE 0 END) AS ihg_16,
    SUM(CASE WHEN ihg.cluster_id = 17 THEN 1 ELSE 0 END) AS ihg_17,
    SUM(CASE WHEN ihg.cluster_id = 18 THEN 1 ELSE 0 END) AS ihg_18,
    SUM(CASE WHEN ihg.cluster_id = 19 THEN 1 ELSE 0 END) AS ihg_19,
    SUM(CASE WHEN ihg.cluster_id = 20 THEN 1 ELSE 0 END) AS ihg_20,
    SUM(CASE WHEN ihg.cluster_id = 21 THEN 1 ELSE 0 END) AS ihg_21,
    SUM(CASE WHEN ihg.cluster_id = 22 THEN 1 ELSE 0 END) AS ihg_22,
    SUM(CASE WHEN ihg.cluster_id = 23 THEN 1 ELSE 0 END) AS ihg_23,
    SUM(CASE WHEN ihg.cluster_id = 24 THEN 1 ELSE 0 END) AS ihg_24,
    SUM(CASE WHEN ihg.cluster_id = 25 THEN 1 ELSE 0 END) AS ihg_25,
    SUM(CASE WHEN ihg.cluster_id = 26 THEN 1 ELSE 0 END) AS ihg_26,
    SUM(CASE WHEN ihg.cluster_id = 27 THEN 1 ELSE 0 END) AS ihg_27,
    SUM(CASE WHEN ihg.cluster_id = 28 THEN 1 ELSE 0 END) AS ihg_28,
    SUM(CASE WHEN ihg.cluster_id = 29 THEN 1 ELSE 0 END) AS ihg_29,
    SUM(CASE WHEN ihg.cluster_id = 30 THEN 1 ELSE 0 END) AS ihg_30,
    SUM(CASE WHEN ihg.cluster_id = 31 THEN 1 ELSE 0 END) AS ihg_31,
    SUM(CASE WHEN ihg.cluster_id = 32 THEN 1 ELSE 0 END) AS ihg_32,
    SUM(CASE WHEN ihg.cluster_id = 33 THEN 1 ELSE 0 END) AS ihg_33,
    SUM(CASE WHEN ihg.cluster_id = 34 THEN 1 ELSE 0 END) AS ihg_34,
    SUM(CASE WHEN ihg.cluster_id = 35 THEN 1 ELSE 0 END) AS ihg_35,
    SUM(CASE WHEN ihg.cluster_id = 36 THEN 1 ELSE 0 END) AS ihg_36,
    SUM(CASE WHEN ihg.cluster_id = 37 THEN 1 ELSE 0 END) AS ihg_37,
    SUM(CASE WHEN ihg.cluster_id = 38 THEN 1 ELSE 0 END) AS ihg_38,
    SUM(CASE WHEN ihg.cluster_id = 39 THEN 1 ELSE 0 END) AS ihg_39,
    SUM(CASE WHEN ihg.cluster_id = 40 THEN 1 ELSE 0 END) AS ihg_40,
    SUM(CASE WHEN ihg.cluster_id = 41 THEN 1 ELSE 0 END) AS ihg_41,
    SUM(CASE WHEN ihg.cluster_id = 42 THEN 1 ELSE 0 END) AS ihg_42,
    SUM(CASE WHEN ihg.cluster_id = 43 THEN 1 ELSE 0 END) AS ihg_43,
    SUM(CASE WHEN ihg.cluster_id = 44 THEN 1 ELSE 0 END) AS ihg_44,
    SUM(CASE WHEN ihg.cluster_id = 45 THEN 1 ELSE 0 END) AS ihg_45,
    SUM(CASE WHEN ihg.cluster_id = 46 THEN 1 ELSE 0 END) AS ihg_46,
    SUM(CASE WHEN ihg.cluster_id = 47 THEN 1 ELSE 0 END) AS ihg_47,
    SUM(CASE WHEN ihg.cluster_id = 48 THEN 1 ELSE 0 END) AS ihg_48,
    SUM(CASE WHEN ihg.cluster_id = 49 THEN 1 ELSE 0 END) AS ihg_49,
    SUM(CASE WHEN ihg.cluster_id = 50 THEN 1 ELSE 0 END) AS ihg_50,
    SUM(CASE WHEN ihg.cluster_id = 51 THEN 1 ELSE 0 END) AS ihg_51,
    SUM(CASE WHEN ihg.cluster_id = 52 THEN 1 ELSE 0 END) AS ihg_52,
    SUM(CASE WHEN ihg.cluster_id = 53 THEN 1 ELSE 0 END) AS ihg_53,
    SUM(CASE WHEN ihg.cluster_id = 54 THEN 1 ELSE 0 END) AS ihg_54,
    SUM(CASE WHEN ihg.cluster_id = 55 THEN 1 ELSE 0 END) AS ihg_55,
    SUM(CASE WHEN ihg.cluster_id = 56 THEN 1 ELSE 0 END) AS ihg_56, 
    SUM(CASE WHEN ihg.cluster_id =  57 THEN 1 ELSE 0 END) AS ihg_57 ,
    SUM(CASE WHEN ihg.cluster_id =  58 THEN 1 ELSE 0 END) AS ihg_58 ,
    SUM(CASE WHEN ihg.cluster_id =  59 THEN 1 ELSE 0 END) AS ihg_59 ,
    SUM(CASE WHEN ihg.cluster_id =  60 THEN 1 ELSE 0 END) AS ihg_60 ,
    SUM(CASE WHEN ihg.cluster_id =  61 THEN 1 ELSE 0 END) AS ihg_61 ,
    SUM(CASE WHEN ihg.cluster_id =  62 THEN 1 ELSE 0 END) AS ihg_62 ,
    SUM(CASE WHEN ihg.cluster_id =  63 THEN 1 ELSE 0 END) AS ihg_63 ,
    SUM(CASE WHEN ihg.cluster_id =  64 THEN 1 ELSE 0 END) AS ihg_64 ,
    SUM(CASE WHEN ihg.cluster_id =  65 THEN 1 ELSE 0 END) AS ihg_65 ,
    SUM(CASE WHEN ihg.cluster_id =  66 THEN 1 ELSE 0 END) AS ihg_66 ,
    SUM(CASE WHEN ihg.cluster_id =  67 THEN 1 ELSE 0 END) AS ihg_67 ,
    SUM(CASE WHEN ihg.cluster_id =  68 THEN 1 ELSE 0 END) AS ihg_68 ,
    SUM(CASE WHEN ihg.cluster_id =  69 THEN 1 ELSE 0 END) AS ihg_69 ,
    SUM(CASE WHEN ihg.cluster_id =  70 THEN 1 ELSE 0 END) AS ihg_70 ,
    SUM(CASE WHEN ihg.cluster_id =  71 THEN 1 ELSE 0 END) AS ihg_71 ,
    SUM(CASE WHEN ihg.cluster_id =  72 THEN 1 ELSE 0 END) AS ihg_72 ,
    SUM(CASE WHEN ihg.cluster_id =  73 THEN 1 ELSE 0 END) AS ihg_73 ,
    SUM(CASE WHEN ihg.cluster_id =  74 THEN 1 ELSE 0 END) AS ihg_74 ,
    SUM(CASE WHEN ihg.cluster_id =  75 THEN 1 ELSE 0 END) AS ihg_75 ,
    SUM(CASE WHEN ihg.cluster_id =  76 THEN 1 ELSE 0 END) AS ihg_76 ,
    SUM(CASE WHEN ihg.cluster_id =  77 THEN 1 ELSE 0 END) AS ihg_77 ,
    SUM(CASE WHEN ihg.cluster_id =  78 THEN 1 ELSE 0 END) AS ihg_78 ,
    SUM(CASE WHEN ihg.cluster_id =  79 THEN 1 ELSE 0 END) AS ihg_79 ,
    SUM(CASE WHEN ihg.cluster_id =  80 THEN 1 ELSE 0 END) AS ihg_80 ,
    SUM(CASE WHEN ihg.cluster_id =  81 THEN 1 ELSE 0 END) AS ihg_81 ,
    SUM(CASE WHEN ihg.cluster_id =  82 THEN 1 ELSE 0 END) AS ihg_82 ,
    SUM(CASE WHEN ihg.cluster_id =  83 THEN 1 ELSE 0 END) AS ihg_83 ,
    SUM(CASE WHEN ihg.cluster_id =  84 THEN 1 ELSE 0 END) AS ihg_84 ,
    SUM(CASE WHEN ihg.cluster_id =  85 THEN 1 ELSE 0 END) AS ihg_85 ,
    SUM(CASE WHEN ihg.cluster_id =  86 THEN 1 ELSE 0 END) AS ihg_86 ,
    SUM(CASE WHEN ihg.cluster_id =  87 THEN 1 ELSE 0 END) AS ihg_87 ,
    SUM(CASE WHEN ihg.cluster_id =  88 THEN 1 ELSE 0 END) AS ihg_88 ,
    SUM(CASE WHEN ihg.cluster_id =  89 THEN 1 ELSE 0 END) AS ihg_89 ,
    SUM(CASE WHEN ihg.cluster_id =  90 THEN 1 ELSE 0 END) AS ihg_90 ,
    SUM(CASE WHEN ihg.cluster_id =  91 THEN 1 ELSE 0 END) AS ihg_91 ,
    SUM(CASE WHEN ihg.cluster_id =  92 THEN 1 ELSE 0 END) AS ihg_92 ,
    SUM(CASE WHEN ihg.cluster_id =  93 THEN 1 ELSE 0 END) AS ihg_93 ,
    SUM(CASE WHEN ihg.cluster_id =  94 THEN 1 ELSE 0 END) AS ihg_94 ,
    SUM(CASE WHEN ihg.cluster_id =  95 THEN 1 ELSE 0 END) AS ihg_95 ,
    SUM(CASE WHEN ihg.cluster_id =  96 THEN 1 ELSE 0 END) AS ihg_96 ,
    SUM(CASE WHEN ihg.cluster_id =  97 THEN 1 ELSE 0 END) AS ihg_97 ,
    SUM(CASE WHEN ihg.cluster_id =  98 THEN 1 ELSE 0 END) AS ihg_98 ,
    SUM(CASE WHEN ihg.cluster_id =  99 THEN 1 ELSE 0 END) AS ihg_99 ,
    SUM(CASE WHEN ihg.cluster_id =  100 THEN 1 ELSE 0 END) AS ihg_100 ,
    SUM(CASE WHEN ihg.cluster_id =  101 THEN 1 ELSE 0 END) AS ihg_101 ,
    SUM(CASE WHEN ihg.cluster_id =  102 THEN 1 ELSE 0 END) AS ihg_102 ,
    SUM(CASE WHEN ihg.cluster_id =  103 THEN 1 ELSE 0 END) AS ihg_103 ,
    SUM(CASE WHEN ihg.cluster_id =  104 THEN 1 ELSE 0 END) AS ihg_104 ,
    SUM(CASE WHEN ihg.cluster_id =  105 THEN 1 ELSE 0 END) AS ihg_105 ,
    SUM(CASE WHEN ihg.cluster_id =  106 THEN 1 ELSE 0 END) AS ihg_106 ,
    SUM(CASE WHEN ihg.cluster_id =  107 THEN 1 ELSE 0 END) AS ihg_107 ,
    SUM(CASE WHEN ihg.cluster_id =  108 THEN 1 ELSE 0 END) AS ihg_108 ,
    SUM(CASE WHEN ihg.cluster_id =  109 THEN 1 ELSE 0 END) AS ihg_109 ,
    SUM(CASE WHEN ihg.cluster_id =  110 THEN 1 ELSE 0 END) AS ihg_110 ,
    SUM(CASE WHEN ihg.cluster_id =  111 THEN 1 ELSE 0 END) AS ihg_111 ,
    SUM(CASE WHEN ihg.cluster_id =  112 THEN 1 ELSE 0 END) AS ihg_112 ,
    SUM(CASE WHEN ihg.cluster_id =  113 THEN 1 ELSE 0 END) AS ihg_113 ,
    SUM(CASE WHEN ihg.cluster_id =  114 THEN 1 ELSE 0 END) AS ihg_114 ,
    SUM(CASE WHEN ihg.cluster_id =  115 THEN 1 ELSE 0 END) AS ihg_115 ,
    SUM(CASE WHEN ihg.cluster_id =  116 THEN 1 ELSE 0 END) AS ihg_116 ,
    SUM(CASE WHEN ihg.cluster_id =  117 THEN 1 ELSE 0 END) AS ihg_117 ,
    SUM(CASE WHEN ihg.cluster_id =  118 THEN 1 ELSE 0 END) AS ihg_118 ,
    SUM(CASE WHEN ihg.cluster_id =  119 THEN 1 ELSE 0 END) AS ihg_119 ,
    SUM(CASE WHEN ihg.cluster_id =  120 THEN 1 ELSE 0 END) AS ihg_120 ,
    SUM(CASE WHEN ihg.cluster_id =  121 THEN 1 ELSE 0 END) AS ihg_121 ,
    SUM(CASE WHEN ihg.cluster_id =  122 THEN 1 ELSE 0 END) AS ihg_122 ,
    SUM(CASE WHEN ihg.cluster_id =  123 THEN 1 ELSE 0 END) AS ihg_123 ,
    SUM(CASE WHEN ihg.cluster_id =  124 THEN 1 ELSE 0 END) AS ihg_124 ,
    SUM(CASE WHEN ihg.cluster_id =  125 THEN 1 ELSE 0 END) AS ihg_125 ,
    SUM(CASE WHEN ihg.cluster_id =  126 THEN 1 ELSE 0 END) AS ihg_126 ,
    SUM(CASE WHEN ihg.cluster_id =  127 THEN 1 ELSE 0 END) AS ihg_127 
    
FROM 
    SegmentOct20 so
JOIN 
    ImagesHandsPositions ihp ON ihp.image_id = so.image_id
JOIN 
    ImagesHandsGestures ihg ON ihg.image_id = so.image_id
JOIN 
    ImagesTopics it ON it.image_id = so.image_id
WHERE it.topic_id = {topic_id}
GROUP BY 
    ihp.cluster_id
ORDER BY 
    ihp_cluster;
"""

# Loop through each topic_id (0 to 63) and save results to CSV
for topic_id in range(64):
    query = sql_query_template.format(topic_id=topic_id)

    print(f"about to query for topic_id {topic_id}")

    # Execute query and fetch data into a DataFrame
    df = pd.read_sql(query, engine)
    
    # Define file name for the CSV
    csv_file_path = f"{ROOT_FOLDER_PATH}/topic_{topic_id}.csv"
    
    # Save to CSV
    df.to_csv(csv_file_path, index=False)
    print(f"Saved results for topic_id {topic_id} to {csv_file_path}")

# Close the session
session.close()
