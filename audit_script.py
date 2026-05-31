import os
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool
from mp_db_io import DataIO

helper_table = 'SegmentHelper_T11_Oct20_COCO_Custom_every40'
class_id = 41
slots = ['feet_object_id','left_hand_object_id','right_hand_object_id']

io = DataIO()
db = io.db
engine = create_engine(
    'mysql+pymysql://{user}:{pw}@/{db}?unix_socket={socket}'.format(
        user=db['user'], pw=db['pass'], db=db['name'], socket=db['unix_socket']
    ),
    poolclass=NullPool,
)

out_dir = 'analysis/imagesdetections_debug/object_placement_audit'
os.makedirs(out_dir, exist_ok=True)

all_rows = []
counts = []

with engine.connect() as conn:
    # sanity
    helper_cols = [r[0] for r in conn.execute(text(f"SHOW COLUMNS FROM {helper_table}")).fetchall()]
    print('helper columns:', helper_cols)

    for slot_col in slots:
        q_count = text(f'SELECT COUNT(*) FROM ImagesDetections idt JOIN {helper_table} sh ON sh.image_id = idt.image_id JOIN Detections d ON d.detection_id = idt.{slot_col} WHERE d.class_id = :cid')
        cnt = conn.execute(q_count, {'cid': class_id}).scalar()
        counts.append({'slot': slot_col, 'count': int(cnt or 0)})

        q_sample = text(f'''
            SELECT 
                idt.image_id, 
                :slot AS slot,
                idt.{slot_col} AS detection_id, 
                d.conf, 
                d.bbox_norm, 
                i.imagename, 
                i.site_name_id, 
                i.contentUrl
            FROM ImagesDetections idt
            JOIN {helper_table} sh ON sh.image_id = idt.image_id
            JOIN Detections d ON d.detection_id = idt.{slot_col}
            LEFT JOIN Images i ON i.image_id = idt.image_id
            WHERE d.class_id = :cid
            ORDER BY idt.{slot_col} DESC
            LIMIT 20
        ''')
        rows = conn.execute(q_sample, {'cid': class_id, 'slot': slot_col}).fetchall()
        all_rows.extend([dict(r._mapping) for r in rows])

counts_df = pd.DataFrame(counts)
samples_df = pd.DataFrame(all_rows)

counts_path = os.path.join(out_dir, 'cup41_slot_counts.csv')
samples_path = os.path.join(out_dir, 'cup41_spotcheck_samples_fast.csv')

counts_df.to_csv(counts_path, index=False)
samples_df.to_csv(samples_path, index=False)

print('counts:')
print(counts_df.to_string(index=False))
print('saved:', counts_path)
print('saved:', samples_path)
print('preview:')
if not samples_df.empty:
    print(samples_df.head(5).to_string(index=False))
else:
    print("No matches found.")
