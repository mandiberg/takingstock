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
from sqlalchemy import create_engine, MetaData, select, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool



# Class to handle BSON export tasks
class MongoBSONExporter:
    def __init__(self, mongo_db):
        self.mongo_db = mongo_db

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

        # setting collection objects from dict
        for collection_name in self.collection_names:
            globals()[f"self.{collection_name}_collection"] = mongo_db[collection_name]

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
        # print(f" first_key = {first_key}, type = {type(first_key)}")
        # print(f" batch_bson[first_key] = {batch_bson[first_key]}, type = {type(batch_bson[first_key])}")

        if isinstance(first_key, int) and isinstance(batch_bson[first_key], dict):
        # if 'encoding_id' in batch_bson[first_key].keys():
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

    def read_and_store_bson_batch(self, engine, mongo_db, batch_list, passed_collection, table_name="compare_sql_mongo_results", ):
        writing_individual_files = False
        DO_ONLY_ONE = True
        for file in batch_list:
            print(f"Processing BSON file: {file}")
            try:
                docs = self.read_bson(file)
            except Exception as e:
                print(f"Error reading BSON file {file}: {e}")
                continue
            # print(f"Opened BSON file: {docs}")
            print(f"collection before if  = {passed_collection}")
            if passed_collection is None:
                filename = os.path.basename(file)
                # look for self.col_to_collection.keys() in filename to determine collection
                # print(f"filename = {filename}")
                for key in self.collection_names:
                    # print(f"checking key: {key} in filename")
                    if key in filename:
                        # collection = self.col_to_collection[key]                        
                        collection = key
                        # collection = f"self.{collection_name}_collection"
                        writing_individual_files = True
                        # print(f"Determined collection as {collection} from filename {file}")
                        break
                # if collection is still None, use the first part of the filename before the first underscore
                if collection is None:
                    collection = filename.split("_")[0]
                    writing_individual_files = True

                print(f"Determined collection as {collection} from filename {file}")
            else:
                collection = passed_collection

            this_field_list = self.document_names_dict[collection]
            print(f"read {len(docs)} documents from BSON file {file} to write to {table_name}")
            for doc in docs:
                if not bool(doc):
                    # print(f"Skipping None document in file {file}")
                    continue
                image_id = doc.get("image_id", None)
                encoding_id = doc.get("encoding_id", None)
                # print(f"Processing document with image_id {image_id} encoding_id {encoding_id}")
                if not image_id and encoding_id is not None:
                    print(f"NO image_id but has encoding_id {encoding_id} in file {file}") # but has encoding_id
                if not image_id and not encoding_id:
                    print(f"Skipping document without image_id or encoding_id in file {file}: {doc}")
                    continue
                # check for each value in this_field_list in the doc
                for field in this_field_list:
                    if field in doc:
                        collection = self.col_to_collection.get(field, None)

                        # print(f"Found field {collection}:{field} len: {len(doc[field])} in document for image_id {image_id} encoding_id {encoding_id}")
                        # continue
                        if collection:
                            # pass in encoding_id. if is None, it will be handled by write_Mongo_value
                            success = self.write_Mongo_value(engine, mongo_db, collection, "image_id", image_id, field, doc[field], encoding_id)
                            if not success:
                                print(f"Failed to write Mongo value for image_id {image_id}, field {field}")
                            else:
                                if encoding_id is None:
                                    encoding_id = self.lookup_encoding_id(engine, image_id)
                                # print(f"Updating encoding_id {encoding_id} setting {field} to NULL")
                                self.write_MySQL_value(engine, table_name, "encoding_id", encoding_id, field, "NULL")

            # after finishing the file, save as completed...
            if writing_individual_files:
                # print("writing individual files is true, saving file to BsonFileLog")
                # Insert file to completed_bson_file field in BsonFileLog table.
                stmt = f"INSERT INTO BsonFileLog (completed_bson_file) VALUES ('{file}');"
                try:
                    with engine.connect() as connection:
                        connection.execute(text(stmt))
                        connection.commit()
                except Exception as e:
                    print(f"Error updating BsonFileLog: {e}")
                print(f"Completed and wrote to BsonFileLog: {filename}")

    def lookup_encoding_id(self, engine, image_id):
        stmt = f"SELECT encoding_id FROM Encodings WHERE image_id = {image_id};"
        # print(f"Retrieving encoding_id for image_id {image_id} using query: {stmt}")
        try:
            with engine.connect() as connection:
                result = connection.execute(text(stmt))
                encoding_id = result.scalar()
                # print("encoding_id", encoding_id)
        except Exception as e:
            print(f"Error retrieving encoding_id: {e}")
        return encoding_id


    def build_folder_bson_file_list(self, export_dir):
        list_of_bson_files = [f for f in os.listdir(export_dir) if f.endswith('.bson')]
        return list_of_bson_files

    def build_folder_bson_file_list_full_paths(self, export_dir, batch_size=8):
        list_of_bson_files = self.build_folder_bson_file_list(export_dir)
        list_of_bson_files_full_paths = [os.path.join(export_dir, f) for f in list_of_bson_files]
        all_batches = self.split_into_batches(batch_size, list_of_bson_files_full_paths)
        return all_batches

    def split_into_batches(self, batch_size, list_of_bson_files):
        all_batches = []
        for i in range(0, len(list_of_bson_files), batch_size):
            this_batch = list_of_bson_files[i:i + batch_size]
            all_batches.append(this_batch)
        return all_batches

    def build_batch_list(self, export_dir, batch_size):
        list_of_bson_files = self.build_folder_bson_file_list(export_dir)
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
            all_batches = self.split_into_batches(batch_size, collection_bson_files)
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

    def write_Mongo_value(self, engine, mongo_db, collection_name, key, id, col, cell_value, mongo_encoding_id=None):
        if key and id and col:
            collection = mongo_db[collection_name]
            mongo_results = collection.find_one({"image_id": id})
            if mongo_results[col] is not None and mongo_results[col] == cell_value:
                if mongo_results["encoding_id"] is not None and mongo_results["encoding_id"] % 100 == 0:
                    print(f" === processed {collection_name} up to encoding {mongo_results['encoding_id']} as {col} is already up to date.")
                return True
            # print(f"this col results = {mongo_results[col]}", col)
            if collection_name == "encodings" and key == "image_id":
                # deal with None values, which are also often values where the encoding_id is wrong in mongo
                if mongo_encoding_id is not None:
                    # FOR REDOING THE RESHARD TO CATCH NONES
                    # print(f"have a eid: {mongo_encoding_id} so going to continue as it probably wrote correctly last time")
                    # return False

                    # FOR REGULAR --- >>>>
                    # print("got an mongo_encoding_id", mongo_encoding_id)
                    query = {key: id, "encoding_id": mongo_encoding_id}
                else:
                    # print(f"this col results = {mongo_results['encoding_id']}")
                    # print(f"No encoding_id provided for image_id {id}, looking up encoding_id from Mongo and MySQL")
                    # mongo_encoding_id_results = collection.find_one({"image_id": id}, {"encoding_id": 1})
                    if mongo_results and "encoding_id" in mongo_results:
                        # print(f"mongo_results = {mongo_results}")
                        mongo_encoding_id = mongo_results["encoding_id"]
                        # print(f"Found image_id {id} with mongo_encoding_id {mongo_encoding_id}")
                        print(f"Found image_id {id} with mongo_encoding_id {mongo_encoding_id}")
                    else:
                        print(f" ~ Could not find mongo_encoding_id for image_id {id} in MongoDB collection {collection_name}")
                        mongo_encoding_id = None
                    # if no encoding_id use engine to get corect encoding_id from encodings table in mysql
                    # print("going to lookup encoding_id from MySQL from id", id)
                    mysql_encoding_id = self.lookup_encoding_id(engine, id)
                    # print(f"Found image_id {id} with mongo_encoding_id {mongo_encoding_id} and SQL encoding_id: {mysql_encoding_id}")
                    if mongo_encoding_id != mysql_encoding_id:
                        print(f" XXX Mismatch for image_id {id}: mongo_encoding_id {mongo_encoding_id} vs mysql_encoding_id {mysql_encoding_id}")
                        # replace mongo_encoding_id with mysql_encoding_id in the mongodb collection
                        replace_encoding_id_query = {key: id, "encoding_id": mongo_encoding_id}
                        replace_encoding_id_update = {"$set": {"encoding_id": mysql_encoding_id}}
                        # print("going to do this:", replace_encoding_id_query, replace_encoding_id_update)
                        result = collection.update_one(replace_encoding_id_query, replace_encoding_id_update)
                        # print("result is ", result)
                        # query = {key: id, "encoding_id": mysql_encoding_id}
                    else:
                        print(f" --- no mismatch for image_id {id}: mongo_encoding_id {mongo_encoding_id} vs mysql_encoding_id {mysql_encoding_id}")
                    query = {key: id, "encoding_id": mysql_encoding_id}

            else:
                query = {key: id}
            update = {"$set": {col: cell_value}}
            try:
                # print(f"Upserting into Mongo collection {collection_name} for {query} setting {col} to value of length {len(str(cell_value)) if cell_value else 0}")
                result = collection.update_one(query, update, upsert=True)
                if result.matched_count > 0 or result.upserted_id is not None:
                    # print(f"Success for {id} {key} set {col} to {str(cell_value)[:20]}")
                    return True
                else:
                    print(f"No document found or upserted for {query}")
            except Exception as e:
                if "E11000" in str(e):
                    print(f"Duplicate key error: {e}")
                    # isolate the duplicate key value from the error message
                    # and print it
                    duplicate_key_value = str(e).split("key: ")[1].split(" dup key")[0]
                    # print(f"Duplicate key value: {duplicate_key_value}")
                    # isolate image_id from duplicate_key_value that has this pattern: { image_id: 73726720 }
                    if "image_id" in duplicate_key_value:
                        image_id_str = duplicate_key_value.split("image_id: ")[1].split(" }")[0]
                        image_id_int = int(image_id_str)
                        print(f"Found image_id {image_id_int} with mysql_encoding_id {mysql_encoding_id} and mongo_encoding_id: {mongo_encoding_id}")
                else:
                    print(f"Error writing Mongo value: {e}")
                
        else:
            print("Missing parameters for write_Mongo_value")
        return False

    def write_MySQL_value(self, engine, table_name, key, id, col, cell_value):
        # print("writing MySQL")
        if key and id and col:
            from sqlalchemy import text
            stmt = f"UPDATE {table_name} SET {col}={cell_value} WHERE {key} = {id};"
            # print(f"Executing SQL: {stmt}")
            # execute the statement here using your database connection
            # For example, using a SQLAlchemy session:
            try:
                with engine.connect() as connection:
                    connection.execute(text(stmt))
                    connection.commit()
                # print(f"Success for     {id} {key} set {col} to {cell_value}")
            except Exception as e:
                print(f"Error writing MySQL value: {e}")
        else:
            print("Missing parameters for write_MySQL_value")
        


