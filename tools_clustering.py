from sqlalchemy import create_engine, select, Column, Integer, Float, ForeignKey, BLOB
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool
from my_declarative_base import Base, Images
import pickle
import numpy as np

class ToolsClustering:
    """Store key clustering info for use across codebase"""

    def __init__(self, CLUSTER_TYPE, VERBOSE=False):
        self.VERBOSE = VERBOSE
        self.CLUSTER_TYPE = CLUSTER_TYPE
        self.CLUSTER_MEDIANS = None
        self.CLUSTER_DATA = {
            "BodyPoses": {"data_column": "mongo_body_landmarks", "is_feet": 1, "mongo_hand_landmarks": None},
            "BodyPoses3D": {"data_column": "mongo_body_landmarks_3D", "is_feet": 1, "mongo_hand_landmarks": None}, # changed this for testing
            "ArmsPoses3D": {"data_column": "mongo_body_landmarks_3D", "is_feet": None, "mongo_hand_landmarks": 1},
            "HandsGestures": {"data_column": "mongo_hand_landmarks", "is_feet": None, "mongo_hand_landmarks": 1},
            "HandsPositions": {"data_column": "mongo_hand_landmarks_norm", "is_feet": None, "mongo_hand_landmarks": 1},
            "FingertipsPositions": {"data_column": "mongo_hand_landmarks_norm", "is_feet": None, "mongo_hand_landmarks": 1},
            "HSV": {"data_column": ["hue", "sat", "val"], "is_feet": None},
}

    def get_cluster_medians(self, session, Clusters, USE_SUBSET_MEDIANS=False, SUBSET_LANDMARKS=None):
        print("getting cluster medians")
        # Create a SQLAlchemy select statement
        select_query = select(Clusters.cluster_id, Clusters.cluster_median)

        # Execute the query using your SQLAlchemy session
        results = session.execute(select_query)
        median_dict = {}

        # Process the results as needed
        # print(f'Found {len(results)} clusters with medians.')
        # print("results is a sqlalchemy.engine.result.ChunkedIteratorResult object at 0x3233d5400. this is how many: ",len(results))
        for row in results:
            # print(row)
            cluster_id, cluster_median_pickle = row
            # print("cluster_id: ",cluster_id)

            # import pprint
            # pp = pprint.PrettyPrinter(indent=4)
            # print the pickle using pprint
            # pp.pprint(cluster_median_pickle)

            cluster_median = pickle.loads(cluster_median_pickle)
            # print(f"cluster_median {cluster_id}: ", cluster_median)
            # Check the type and content of the deserialized object
            if not isinstance(cluster_median, np.ndarray):
                print(f"Deserialized object is not a numpy array, it's of type: {type(cluster_median)}")

            if USE_SUBSET_MEDIANS:
                # handles body lms subsets
                # print("handling body lms subsets in get_cluster_medians for these subset landmarks: ",sort.SUBSET_LANDMARKS)
                subset_cluster_median = []
                for i in range(len(cluster_median)):
                    # print("i: ",i)
                    if i in SUBSET_LANDMARKS:
                        # print("adding i: ",i)
                        subset_cluster_median.append(cluster_median[i])
                cluster_median = subset_cluster_median
                print(f"subset cluster median {cluster_id}: ",cluster_median)
            median_dict[cluster_id] = cluster_median
        # print("median dict: ",median_dict)
        self.CLUSTER_MEDIANS = median_dict
        return median_dict 

    def get_meta_cluster_dict(self, session, ClustersMetaClusters):
        print("getting cluster medians")
        # Create a SQLAlchemy select statement
        select_query = select(ClustersMetaClusters.cluster_id, ClustersMetaClusters.meta_cluster_id)

        # Execute the query using your SQLAlchemy session
        results = session.execute(select_query)
        meta_cluster_dict = {}

        # Process the results as needed
        # print(f'Found {len(results)} clusters with medians.')
        # print("results is a sqlalchemy.engine.result.ChunkedIteratorResult object at 0x3233d5400. this is how many: ",len(results))
        for row in results:
            # print(row)
            cluster_id, meta_cluster_id = row
            meta_cluster_dict[cluster_id] = meta_cluster_id

        return meta_cluster_dict 


    def set_table_cluster_type(self, META):
        self.META = META
            # set cluster_table_type for ArmsPoses3D, so it pulls from BodyPoses3D table
        # this allows the ArmsPoses3D value to set the Dict and subset landmarks.
        if self.CLUSTER_TYPE == "ArmsPoses3D":
            table_cluster_type = self.CLUSTER_TYPE
        else:
            if self.META:table_cluster_type = "BodyPoses3D"
            else:table_cluster_type = self.CLUSTER_TYPE
        return table_cluster_type
    
    def construct_table_classes(self, table_cluster_type):
        # handle the table objects based on cl.CLUSTER_TYPE
        ClustersTable_name = table_cluster_type
        ImagesClustersTable_name = "Images"+table_cluster_type
        MetaClustersTable_name = "Meta"+table_cluster_type
        ClustersMetaClustersTable_name = "Clusters"+MetaClustersTable_name

        print("ClustersTable_name: ",ClustersTable_name, " ImagesClustersTable_name: ",ImagesClustersTable_name, " MetaClustersTable_name: ",MetaClustersTable_name, " ClustersMetaClustersTable_name: ",ClustersMetaClustersTable_name)

        class Clusters(Base):
            # this doubles as MetaClusters
            __tablename__ = ClustersTable_name

            cluster_id = Column(Integer, primary_key=True, autoincrement=True)
            cluster_median = Column(BLOB)

        class ImagesClusters(Base):
            __tablename__ = ImagesClustersTable_name

            image_id = Column(Integer, ForeignKey(Images.image_id, ondelete="CASCADE"), primary_key=True)
            cluster_id = Column(Integer, ForeignKey(f'{ClustersTable_name}.cluster_id', ondelete="CASCADE"))
            cluster_dist = Column(Float)

        class MetaClusters(Base):
            __tablename__ = MetaClustersTable_name

            cluster_id = Column(Integer, primary_key=True, autoincrement=True)
            cluster_median = Column(BLOB)

        class ClustersMetaClusters(Base):
            __tablename__ = ClustersMetaClustersTable_name
            # cluster_id is pkey, and meta_cluster_id will appear multiple times for multiple clusters
            cluster_id = Column(Integer, ForeignKey(f'{ClustersTable_name}.cluster_id', ondelete="CASCADE"), primary_key=True)
            meta_cluster_id = Column(Integer, ForeignKey(f'{MetaClustersTable_name}.cluster_id', ondelete="CASCADE"))
            cluster_dist = Column(Float)
        return Clusters, ImagesClusters, MetaClusters, ClustersMetaClusters
    
    def set_cluster_metacluster(self, Clusters, ImagesClusters, MetaClusters, ClustersMetaClusters):
        if self.META:
            this_Cluster = MetaClusters
            this_CrosswalkClusters = ClustersMetaClusters
        else:
            this_Cluster = Clusters
            this_CrosswalkClusters = ImagesClusters
        return this_Cluster, this_CrosswalkClusters

    def prep_pose_clusters_enc(self, enc1, median_dict=None):
        # print("prepping pose clusters for enc1: ", enc1, " with median_dict: ", median_dict, " and CLUSTER_MEDIANS: ", self.CLUSTER_MEDIANS)
        if median_dict is None and self.CLUSTER_MEDIANS is not None:
            median_dict = self.CLUSTER_MEDIANS
        # print("current image enc1", enc1)  
        enc1 = np.array(enc1)
        
        this_dist_dict = {}
        for cluster_id in median_dict:
            enc2 = median_dict[cluster_id]
            # print("cluster_id enc2: ", cluster_id,enc2)
            this_dist_dict[cluster_id] = np.linalg.norm(enc1 - enc2, axis=0)
        
        cluster_id, cluster_dist = min(this_dist_dict.items(), key=lambda x: x[1])

        # print(cluster_id)
        return cluster_id, cluster_dist

