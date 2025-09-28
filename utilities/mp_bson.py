'''
This script exports mongo documents to bson files, so they can be imported into the other shard

There are four collections that need to be exported.
The documents include some integers (image_id, encoding_id) alongside pickled data.
The start of the picked data looks like this: Binary.createFromBase64('gASVNwMAAAAAAACMKG1lZGlhcGlwZS5mcmFtZXdvcmsuZm9ybWF0cy5sYW5kbWFya19wYjKUjBZOb3JtYWxpemVkTGFuZG1hcmtM…', 0)
The hand_landmarks collection has a different structure -- it uses proper nested JSON instead of pickled data.

Here are the steps to Export Mongo:
    select image_id and all of the booleans listed below from Encodings_Migration where ANY (is_body, is_face, mongo_hand_landmarks) is true and migrated_Mongo is None
    export the bson data from each collection for those image_ids if the mongo booleans are true. These are the booleans:
        mongo_encodings, mongo_body_landmarks, mongo_face_landmarks correspond to mongo_collection 
        mongo_body_landmarks_norm correspond to bboxnormed_collection 
        mongo_hand_landmarks, mongo_hand_landmarks_norm correspond to mongo_hand_collection
        mongo_body_landmarks_3D correspond to body_world_collection
    set "migrated_Mongo" boolean == 0 after exporting the bson data (this avoids re-exporting the same data, and tells us what needs to be reimported in the new shard)

The number of results will be in the 10Ms, so this will be done in batches.
'''

import os
import bson
import gc
import pymongo
from sqlalchemy import create_engine, MetaData, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool



# Class to handle BSON export tasks
class MongoBSONExporter:
    def __init__(self, mongo_collection, bboxnormed_collection, mongo_hand_collection, body_world_collection):
        self.mongo_collection = mongo_collection
        self.bboxnormed_collection = bboxnormed_collection
        self.mongo_hand_collection = mongo_hand_collection
        self.body_world_collection = body_world_collection

        self.collection_names = ['encodings', 'body_landmarks_norm', 'hand_landmarks', 'body_world_landmarks']
        self.document_names_dict = {
            "encodings": ["face_landmarks", "body_landmarks", "face_encodings68"],
            "body_landmarks_norm": ["nlms"],
            "hand_landmarks": ["left_hand", "right_hand"],
            "body_world_landmarks": ["body_world_landmarks"]
        }
        self.sql_field_names_dict = {
            "face_landmarks": "mongo_face_landmarks",
            "body_landmarks": "mongo_body_landmarks",
            "face_encodings68": "mongo_encodings",
            "nlms": "mongo_body_landmarks_norm",
            "left_hand": "mongo_hand_landmarks",
            "right_hand": "mongo_hand_landmarks",
            "body_world_landmarks": "mongo_body_landmarks_3D"
        }

        # Map document names to their corresponding collection
        self.col_to_collection = {}
        for collection, docnames in self.document_names_dict.items():
            for docname in docnames:
                self.col_to_collection[docname] = collection
        # print("Column to Collection mapping:", self.col_to_collection)

    def export_task(self, row):
        image_id = row.image_id
        bson_data = {}
        # encodings, body_landmarks, face_landmarks
        if any([row.mongo_encodings, row.mongo_body_landmarks, row.mongo_face_landmarks]):
            doc = self.mongo_collection.find_one({"image_id": image_id})
            if doc:
                bson_data['encodings'] = bson.BSON.encode(doc)
        if row.mongo_body_landmarks_norm:
            doc = self.bboxnormed_collection.find_one({"image_id": image_id})
            if doc:
                bson_data['body_landmarks_norm'] = bson.BSON.encode(doc)
        if any([row.mongo_hand_landmarks, row.mongo_hand_landmarks_norm]):
            doc = self.mongo_hand_collection.find_one({"image_id": image_id})
            if doc:
                bson_data['hand_landmarks'] = bson.BSON.encode(doc)
        if row.mongo_body_landmarks_3D:
            doc = self.body_world_collection.find_one({"image_id": image_id})
            if doc:
                bson_data['body_landmarks_3D'] = bson.BSON.encode(doc)
        return image_id, bson_data

    def write_bson_batches(self, batch_bson, offset, export_dir, collections_to_export,verbose=True):
        """
        Write batch BSON to files (one file per collection per batch)
        """
        import os
        # print(f" batch_bson  = {batch_bson}")
        # get first key to see if we are using new or old way
        # print("collections_to_export", collections_to_export)
        first_key = next(iter(batch_bson), None)
        store_encoding_id = False
        # print(f" first_key = {first_key}")
        # print(f" batch_bson[first_key].keys() = {batch_bson[first_key].keys()}")
        # check if 'encoding_id' is in the keys of batch_bson[first_key]
        if 'encoding_id' in batch_bson[first_key].keys():
            batch_dict = batch_bson
            batch_collection_data= {}
            for collection in self.document_names_dict.keys():
                batch_collection_data[collection] = []
            # print(f" NEW WAY for {collections_to_export}")
            # print(f" batch_dict has {len(batch_dict['encoding_id'])} encoding_id entries")
            for image_id, data in batch_dict.items():
                # encoding_id = data.get('encoding_id', None)
                # print(f" Writing data for image_id {image_id} encoding_id {encoding_id}, keys: {list(data.keys())}")
                # print(f"checking data: {data}")
                for collection, docnames in self.document_names_dict.items():
                    this_document_collection_data = {}
                    for key, value in data.items():
                        if key in docnames:
                            # print(f"key {key} in collection {collection} as it's in collections_to_export"  )
                            this_document_collection_data["image_id"] = image_id
                            # print(f"Skipping key {key} in collection {collection} as it's not in collections_to_export")
                            if key in self.document_names_dict["encodings"]:
                                this_document_collection_data["encoding_id"] = data.get("encoding_id", None)
                            this_document_collection_data[key] = value
                    batch_collection_data[collection].append(this_document_collection_data)

            # print(f" batch_collection_data len: {len(batch_collection_data)}")

            for collection, docs in batch_collection_data.items():
                # if docs is an empty list, or a list of empty dicts, skip
                if not docs or all(doc == {} for doc in docs):
                    continue
                batch_file = os.path.join(export_dir, f"{collection}_batch_{offset}.bson")
                with open(batch_file, "ab") as f:  # Use "ab" to append in case of multiple docs
                    for doc in docs:
                        # Ensure all binary fields are BSON Binary
                        for k, v in doc.items():
                            if isinstance(v, bytes):
                                doc[k] = bson.Binary(v)
                        f.write(bson.BSON.encode(doc))

            if verbose:
                print(f"Wrote document for image_id {image_id} to {batch_file}")
        else:
            print(" OLD WAY: batch_bson has no encoding_id entries")
            
            for key in batch_bson:
                # key refers to collection names
                if batch_bson[key]:
                    print(f"Writing {len(batch_bson[key])} documents to {key}_batch_{offset}.bson")
                    batch_file = os.path.join(export_dir, f"{key}_batch_{offset}.bson")
                    with open(batch_file, "wb") as f:
                        for doc_bson in batch_bson[key]:
                            print(f"Writing document '{doc_bson}' to {batch_file}")
                            f.write(doc_bson)
                    if verbose:
                        print(f"Wrote {len(batch_bson[key])} docs to {batch_file}")

    def build_batch_list(self, export_dir, batch_size):
        list_of_bson_files = [f for f in os.listdir(export_dir) if f.endswith('.bson')]
        # print(f"Found {len(list_of_bson_files)} BSON files in {export_dir}")
        collection_files_dict = {}
        for collection_name in self.collection_names:
            collection_bson_files = [f for f in list_of_bson_files if collection_name in f]
            if not collection_bson_files:
                # print(f" -- No BSON files found for collection {collection_name}, skipping")
                continue
            collection_bson_files = [os.path.join(export_dir, f) for f in collection_bson_files]
            # print(f" -- Found {len(collection_bson_files)} BSON files for collection {collection_name}")
            # go through the files in batches of batch_size
            all_batches = []
            for i in range(0, len(collection_bson_files), batch_size):
                this_batch = collection_bson_files[i:i + batch_size]
                all_batches.append(this_batch)
            collection_files_dict[collection_name] = all_batches
        return collection_files_dict

    def read_batch(self, batch_file, verbose=True):
        # use self.read_bson
        # expects a full path, including directories
        all_docs = []
        for file in batch_file:
            # print(f"Reading BSON file {file}")
            # read the bson files in this batch
            docs = self.read_bson(file)
            # print(f"read {len(docs)} documents from BSON file {file}")
            all_docs.extend(docs)
        return all_docs

    def read_bson(self, filepath, verbose=True):
        """
        Read a BSON file and insert its documents into the specified MongoDB collection.
        """
        from bson import decode_all
        with open(filepath, "rb") as f:
            data = f.read()
            docs = decode_all(data)
        if verbose:
            pass
            # print(f"read {len(docs)} documents from {filepath}")
        return docs
    
    
    def update_SQL_mongo_booleans(self, session, Encodings_Migration, image_ids_exported):
        """
        Update migrated_Mongo status for all exported image_ids in this batch
        """
        session.query(Encodings_Migration).filter(
            Encodings_Migration.image_id.in_(image_ids_exported)
        ).update({"migrated_Mongo": 0}, synchronize_session=False)
        session.commit()
