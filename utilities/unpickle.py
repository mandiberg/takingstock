import json
import base64
import pickle

DATA_TYPE = "face_encodings68"

# Step 1: Load the JSON file
with open("/Users/michaelmandiberg/Library/CloudStorage/Dropbox/Mandiberg-Tender/book_assets/pages_32-23/stock.encodings.json", "r") as f:
    data = json.load(f)

# Step 2: Process each item
unpickled_data = []

for item in data:
    base64_str = item[DATA_TYPE]['$binary']['base64']
    
    # Decode base64 to bytes
    pickled_bytes = base64.b64decode(base64_str)
    
    # Unpickle bytes to Python object
    obj = pickle.loads(pickled_bytes)
    
    unpickled_data.append({
        'encoding_id': item['encoding_id'],
        'image_id': item['image_id'],
        DATA_TYPE: obj
    })

# `unpickled_data` now contains normal Python objects

print(unpickled_data[0])  # show the first one
