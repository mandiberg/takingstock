import os
import pickle
import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool
import sys
import time
import multiprocessing as mp
from functools import partial
import logging
from queue import Empty

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import project-specific models
sys.path.insert(1, '/Users/michaelmandiberg/Documents/GitHub/facemap/')
from my_declarative_base import SegmentTable, Encodings, Base

# MongoDB setup
import pymongo

def get_mongo_client():
    """Create and return a MongoDB client"""
    return pymongo.MongoClient("mongodb://localhost:27017/")

def process_batch(task_queue, result_queue, db_config):
    """Worker process function to process batches of encoding records"""
    # Each process gets its own database connections
    mongo_client = get_mongo_client()
    mongo_db = mongo_client["stock"]
    mongo_collection = mongo_db["hand_landmarks"]
    
    # Create database engine for this process
    engine = create_engine(
        f"mysql+pymysql://{db_config['user']}:{db_config['pass']}@/{db_config['name']}?unix_socket={db_config['unix_socket']}",
        poolclass=NullPool
    )
    Session = sessionmaker(bind=engine)
    session = Session()
    
    pid = os.getpid()
    logger.info(f"Worker {pid} started")
    
    try:
        while True:
            try:
                # Get the next batch from the queue with a timeout
                batch = task_queue.get(timeout=5)
                
                if batch is None:  # Poison pill - terminate the process
                    logger.info(f"Worker {pid} received termination signal")
                    break
                
                batch_results = []
                for encoding_id, image_id in batch:
                    # Fetch pickled hand_landmarks from Mongo
                    mongo_doc = mongo_collection.find_one({"image_id": image_id})
                    is_hand_left = False
                    is_hand_right = False
                    
                    hands = ["left_hand", "right_hand"]
                    for hand in hands:
                        if mongo_doc and mongo_doc.get(hand):
                            # Unpickle to MediaPipe object if needed
                            hand_landmarks = mongo_doc.get(hand)
                            is_hand = True
                            logger.debug(f"{image_id} hand: {hand}, is_hand: {is_hand}")
                        else:
                            is_hand = False
                            logger.debug(f"{image_id} hand: {hand}, is_hand: {is_hand}")
                        
                        if hand == "left_hand":
                            is_hand_left = is_hand
                        elif hand == "right_hand":
                            is_hand_right = is_hand
                    
                    # Store results to be committed in bulk
                    batch_results.append((image_id, is_hand_left, is_hand_right))
                
                # Bulk update in a single transaction
                for image_id, is_hand_left, is_hand_right in batch_results:
                    session.query(Encodings).\
                        filter(Encodings.image_id == image_id).\
                        update({"is_hand_left": is_hand_left, "is_hand_right": is_hand_right})
                
                session.commit()
                
                # Report the maximum ID processed in this batch
                if batch:
                    max_id = max(encoding_id for encoding_id, _ in batch)
                    result_queue.put(max_id)
                    logger.info(f"Worker {pid} processed batch with max ID {max_id}")
                
            except Empty:
                # Queue timeout - check if we should continue
                logger.debug(f"Worker {pid} queue timeout - checking for more work")
                continue
                
    except Exception as e:
        logger.error(f"Worker {pid} error: {str(e)}", exc_info=True)
    finally:
        # Clean up resources
        session.close()
        mongo_client.close()
        engine.dispose()
        logger.info(f"Worker {pid} shutting down")

def main():
    start_time = time.time()
    
    # Get database config
    from mp_db_io import DataIO
    io = DataIO()
    db = io.db
    
    # Set up process pools and queues
    num_processes = mp.cpu_count() - 1 or 1  # Leave one CPU free for system processes
    task_queue = mp.Queue()
    result_queue = mp.Queue()
    
    logger.info(f"Starting pool with {num_processes} worker processes")
    
    # Create and start worker processes
    processes = []
    for _ in range(num_processes):
        p = mp.Process(
            target=process_batch, 
            args=(task_queue, result_queue, db)
        )
        p.start()
        processes.append(p)
    
    # Main process handles database connection for fetching records only
    engine = create_engine(
        f"mysql+pymysql://{db['user']}:{db['pass']}@/{db['name']}?unix_socket={db['unix_socket']}",
        poolclass=NullPool
    )
    Session = sessionmaker(bind=engine)
    session = Session()
    
    # Batch processing parameters
    batch_size = 1000 * num_processes  # Scale batch size with number of processes
    batch_per_worker = 1000  # Records per worker
    last_id = 0
    total_processed = 0
    
    try:
        while True:
            # Fetch next batch from database
            results = (
                session.query(Encodings.encoding_id, Encodings.image_id)
                .filter(
                    Encodings.mongo_hand_landmarks.is_(True),
                    Encodings.is_feet.is_(None),
                    Encodings.encoding_id > last_id
                )
                .order_by(Encodings.encoding_id)
                .limit(batch_size)
                .all()
            )
            
            if not results:
                logger.info("No more rows to process. Exiting.")
                break
            
            # Split results into smaller batches for workers
            num_results = len(results)
            logger.info(f"Fetched {num_results} records to process")
            
            # Create sub-batches for each worker process
            for i in range(0, num_results, batch_per_worker):
                sub_batch = results[i:i+batch_per_worker]
                if sub_batch:
                    task_queue.put(sub_batch)
            
            # Wait for all batches to be processed and get the highest ID
            processed_ids = []
            for _ in range((num_results + batch_per_worker - 1) // batch_per_worker):
                try:
                    max_id = result_queue.get(timeout=60)  # Allow up to 60 seconds per batch
                    processed_ids.append(max_id)
                except Empty:
                    logger.warning("Timeout waiting for batch results")
            
            if processed_ids:
                last_id = max(processed_ids)
                total_processed += num_results
                logger.info(f"Processed up to encoding_id = {last_id}, total processed: {total_processed}")
            else:
                logger.warning("No IDs processed in this batch, trying next batch")
                # Move to next set if no progress was made
                if results:
                    last_id = results[-1][0]
    
    except Exception as e:
        logger.error(f"Main process error: {str(e)}", exc_info=True)
    
    finally:
        # Send termination signal to all worker processes
        for _ in range(num_processes):
            task_queue.put(None)
        
        # Wait for all processes to finish
        for p in processes:
            p.join()
        
        # Clean up resources
        session.close()
        engine.dispose()
        
        elapsed_time = time.time() - start_time
        logger.info(f"Processing complete. Total time: {elapsed_time:.2f} seconds.")
        logger.info(f"Total records processed: {total_processed}")

if __name__ == "__main__":
    main()