from sqlalchemy import Column, Integer,Float, String, Date, Boolean, DECIMAL, BLOB, ForeignKey, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Ethnicity(Base):
    __tablename__ = 'ethnicity'
    ethnicity_id = Column(Integer, primary_key=True, autoincrement=True)
    ethnicity = Column(String(40))

class Gender(Base):
    __tablename__ = 'gender'
    gender_id = Column(Integer, primary_key=True, autoincrement=True)
    gender = Column(String(20))

class Age(Base):
    __tablename__ = 'age'
    age_id = Column(Integer, primary_key=True, autoincrement=True)
    age = Column(String(20))

class AgeDetail(Base):
    __tablename__ = 'agedetail'
    age_detail_id = Column(Integer, primary_key=True, autoincrement=True)
    age_detail = Column(String(20))

class Site(Base):
    __tablename__ = 'site'
    site_name_id = Column(Integer, primary_key=True, autoincrement=True)
    site_name = Column(String(20))

class Location(Base):
    __tablename__ = 'location'
    location_id = Column(Integer, primary_key=True, autoincrement=True)
    location_number_getty = Column(Integer)
    getty_name = Column(String(70))
    nation_name = Column(String(70))
    nation_name_alpha = Column(String(70))
    official_nation_name = Column(String(150))
    sovereignty = Column(String(70))
    code_alpha2 = Column(String(70))
    code_alpha3 = Column(String(70))
    code_numeric = Column(Integer)
    code_iso = Column(String(70))
    
class Images(Base):
    __tablename__ = 'images'
    image_id = Column(Integer, primary_key=True, autoincrement=True)
    site_name_id = Column(Integer, ForeignKey('site.site_name_id'))
    site_image_id = Column(String(50), nullable=False)
    age_id = Column(Integer, ForeignKey('age.age_id'))
    age_detail_id = Column(Integer, ForeignKey('agedetail.age_detail_id'))
    gender_id = Column(Integer, ForeignKey('gender.gender_id'))
    location_id = Column(Integer, ForeignKey('location.location_id'))
    author = Column(String(100))
    caption = Column(String(150))
    contentUrl = Column(String(300), nullable=False)
    description = Column(String(150))
    imagename = Column(String(200))
    uploadDate = Column(Date)

    site = relationship("Site")
    age = relationship("Age")
    agedetail = relationship("AgeDetail")
    gender = relationship("Gender")
    location = relationship("Location")

class Keywords(Base):
    __tablename__ = 'keywords'
    keyword_id = Column(Integer, primary_key=True, autoincrement=True)
    keyword_number = Column(Integer)
    keyword_text = Column(String(50), nullable=False)
    keytype = Column(String(50))
    weight = Column(Integer)
    parent_keyword_id = Column(String(50))
    parent_keyword_text = Column(String(50))

class ImagesKeywords(Base):
    __tablename__ = 'imageskeywords'
    image_id = Column(Integer, ForeignKey('images.image_id'), primary_key=True)
    keyword_id = Column(Integer, ForeignKey('keywords.keyword_id'), primary_key=True)

class ImagesEthnicity(Base):
    __tablename__ = 'imagesethnicity'
    image_id = Column(Integer, ForeignKey('images.image_id'), primary_key=True)
    ethnicity_id = Column(Integer, ForeignKey('ethnicity.ethnicity_id'), primary_key=True)

class Encodings(Base):
    __tablename__ = 'encodings'
    encoding_id = Column(Integer, primary_key=True, autoincrement=True)
    image_id = Column(Integer, ForeignKey('images.image_id'))
    is_face = Column(Boolean)
    is_body = Column(Boolean)
    is_face_distant = Column(Boolean)
    face_x = Column(DECIMAL(6, 3))
    face_y = Column(DECIMAL(6, 3))
    face_z = Column(DECIMAL(6, 3))
    mouth_gap = Column(DECIMAL(6, 3))
    face_landmarks = Column(BLOB)
    bbox = Column(JSON)
    face_encodings = Column(BLOB)
    face_encodings68 = Column(BLOB)
    face_encodings_J3 = Column(BLOB)
    face_encodings_J5 = Column(BLOB)
    face_encodings68_J3 = Column(BLOB)
    face_encodings68_J5 = Column(BLOB)
    body_landmarks = Column(BLOB)

class Clusters(Base):
    __tablename__ = 'Clusters'

    cluster_id = Column(Integer, primary_key=True, autoincrement=True)
    cluster_median = Column(BLOB)

class ImagesClusters(Base):
    __tablename__ = 'ImagesClusters'

    image_id = Column(Integer, ForeignKey('images.image_id'), primary_key=True)
    cluster_id = Column(Integer, ForeignKey('Clusters.cluster_id'))
    
class ClustersTemp(Base):
    __tablename__ = 'ClustersTemp'

    cluster_id = Column(Integer, primary_key=True, autoincrement=True)
    cluster_median = Column(BLOB)

class ImagesClustersTemp(Base):
    __tablename__ = 'ImagesClustersTemp'

    image_id = Column(Integer, ForeignKey('images.image_id'), primary_key=True)
    cluster_id = Column(Integer, ForeignKey('ClustersTemp.cluster_id'))

class BagOfKeywords(Base):
    __tablename__ = 'BagOfKeywords'
    image_id = Column(Integer, primary_key=True, autoincrement=True)
    age_id = Column(Integer, ForeignKey('age.age_id'))
    gender_id = Column(Integer, ForeignKey('gender.gender_id'))
    location_id = Column(Integer, ForeignKey('location.location_id'))
    description = Column(String(150))
    keyword_list = Column(BLOB)  # Pickled list
    tokenized_keyword_list = Column(BLOB)  # Pickled list
    ethnicity_list = Column(BLOB)  # Pickled list

class Topics(Base):
    __tablename__ = 'Topics' 
    topic_id = Column(Integer, primary_key=True, autoincrement=True)
    topic = Column(String(150))
    
class ImagesTopics(Base):
    __tablename__ = 'ImagesTopics' 
    image_id = Column(Integer, ForeignKey('images.image_id'), primary_key=True)
    topic_id = Column(Integer, ForeignKey('Topics.topic_id'))
    topic_score = Column(Float)

class ImagesBG(Base):
    __tablename__ = 'ImagesBG' 
    image_id = Column(Integer, ForeignKey('images.image_id'), primary_key=True)
    #imagename = Column(String(200), ForeignKey('images.imagename'))
    #site_name_id = Column(Integer, ForeignKey('site.site_name_id'))
    hue = Column(Float)
    lum = Column(Float)

# these are for MM use for using segments
# class Clusters(Base):
#     __tablename__ = 'Clusters_May25segment123straight_lessrange'

#     cluster_id = Column(Integer, primary_key=True, autoincrement=True)
#     cluster_median = Column(BLOB)

# class ImagesClusters(Base):
#     __tablename__ = 'ImagesClusters_May25segment123straight_lessrange'

#     image_id = Column(Integer, ForeignKey('images.image_id'), primary_key=True)
#     cluster_id = Column(Integer, ForeignKey('Clusters_May25segment123straight_lessrange.cluster_id'))

