from sqlalchemy import Column, Integer,Float, String, Date, Boolean, DECIMAL, BLOB, ForeignKey, JSON
from sqlalchemy.orm import relationship, declarative_base
#from sqlalchemy.ext.declarative import declarative_base
from sys import platform

Base = declarative_base()

class Ethnicity(Base):
    __tablename__ = 'ethnicity'
    ethnicity_id = Column(Integer, primary_key=True, autoincrement=True)
    ethnicity    = Column(String(40))

class Gender(Base):
    __tablename__ = 'gender'
    gender_id = Column(Integer, primary_key=True, autoincrement=True)
    gender    = Column(String(20))

class Age(Base):
    __tablename__ = 'age'
    age_id = Column(Integer, primary_key=True, autoincrement=True)
    age    = Column(String(20))

class AgeDetail(Base):
    __tablename__ = 'agedetail'
    age_detail_id = Column(Integer, primary_key=True, autoincrement=True)
    age_detail    = Column(String(20))

class Site(Base):
    __tablename__ = 'site'
    site_name_id = Column(Integer, primary_key=True, autoincrement=True)
    site_name    = Column(String(20))

class Model_Release(Base):
    __tablename__ = 'model_release'
    release_name_id = Column(Integer, primary_key=True, autoincrement=True)
    release_name    = Column(String(20))

class Location(Base):
    __tablename__ = 'location'
    location_id           = Column(Integer, primary_key=True, autoincrement=True)
    location_number_getty = Column(Integer)
    getty_name            = Column(String(70))
    nation_name           = Column(String(70))
    nation_name_alpha     = Column(String(70))
    official_nation_name  = Column(String(150))
    sovereignty           = Column(String(70))
    code_alpha2           = Column(String(70))
    code_alpha3           = Column(String(70))
    code_numeric          = Column(Integer)
    code_iso              = Column(String(70))
    
class Images(Base):
    __tablename__ = 'images'
    image_id      = Column(Integer, primary_key=True, autoincrement=True)
    site_name_id  = Column(Integer, ForeignKey('site.site_name_id'))
    site_image_id = Column(String(50), nullable=False)
    age_id        = Column(Integer, ForeignKey('age.age_id'))
    age_detail_id = Column(Integer, ForeignKey('agedetail.age_detail_id'))
    gender_id     = Column(Integer, ForeignKey('gender.gender_id'))
    location_id   = Column(Integer, ForeignKey('location.location_id'))
    author        = Column(String(100))
    caption       = Column(String(150))
    contentUrl    = Column(String(300), nullable=False)
    description   = Column(String(150))
    imagename     = Column(String(200))
    uploadDate    = Column(Date)
    h             = Column(Integer)
    w             = Column(Integer)
    no_image     = Column(Boolean)
    site      = relationship("Site")
    age       = relationship("Age")
    agedetail = relationship("AgeDetail")
    gender    = relationship("Gender")
    location  = relationship("Location")
    if platform == "darwin":
        release_name_id = Column(Integer, ForeignKey('model_release.release_name_id'))
        model_release = relationship("Model_Release")

class WanderingImages(Base):
    __tablename__ = 'wanderingimages'
    wandering_image_id      = Column(Integer, primary_key=True, autoincrement=True)
    wandering_name_site_id  = Column(String(50), nullable=False)
    site_image_id  = Column(String(50))
    site_name_id  = Column(Integer, ForeignKey('site.site_name_id'))

class Keywords(Base):
    __tablename__ = 'keywords'
    keyword_id          = Column(Integer, primary_key=True, autoincrement=True)
    keyword_number      = Column(Integer)
    keyword_text        = Column(String(50), nullable=False)
    keytype             = Column(String(50))
    weight              = Column(Integer)
    parent_keyword_id   = Column(String(50))
    parent_keyword_text = Column(String(50))

class ImagesKeywords(Base):
    __tablename__ = 'imageskeywords'
    image_id   = Column(Integer, ForeignKey('images.image_id'), primary_key=True)
    keyword_id = Column(Integer, ForeignKey('keywords.keyword_id'), primary_key=True)

class ImagesEthnicity(Base):
    __tablename__ = 'imagesethnicity'
    image_id     = Column(Integer, ForeignKey('images.image_id'), primary_key=True)
    ethnicity_id = Column(Integer, ForeignKey('ethnicity.ethnicity_id'), primary_key=True)

class Encodings(Base):
    __tablename__ = 'encodings'
    encoding_id         = Column(Integer, primary_key=True, autoincrement=True)
    image_id            = Column(Integer, ForeignKey('images.image_id'))
    is_face             = Column(Boolean)
    is_body             = Column(Boolean)
    is_face_distant     = Column(Boolean)
    face_x              = Column(DECIMAL(6, 3))
    face_y              = Column(DECIMAL(6, 3))
    face_z              = Column(DECIMAL(6, 3))
    mouth_gap           = Column(DECIMAL(6, 3))
    face_landmarks      = Column(BLOB)
    bbox                = Column(JSON)
    face_encodings      = Column(BLOB)
    face_encodings68    = Column(BLOB)
    face_encodings_J3   = Column(BLOB)
    face_encodings_J5   = Column(BLOB)
    face_encodings68_J3 = Column(BLOB)
    face_encodings68_J5 = Column(BLOB)
    body_landmarks      = Column(BLOB)
    mongo_encodings     = Column(Boolean)
    mongo_body_landmarks   = Column(Boolean)
    mongo_face_landmarks   = Column(Boolean)
    is_small           = Column(Boolean)
    mongo_body_landmarks_norm   = Column(Boolean)
    two_noses              = Column(Boolean)
    is_dupe_of            = Column(Integer, ForeignKey('images.image_id'))
    mongo_hand_landmarks   = Column(Boolean)
    mongo_hand_landmarks_norm   = Column(Boolean)
    is_face_no_lms = Column(Boolean)
    is_feet              = Column(Boolean)

class Encodings_Site2(Base):
    __tablename__ = 'Encodings_Site2'
    encoding_id         = Column(Integer, primary_key=True, autoincrement=True)
    image_id            = Column(Integer, ForeignKey('images.image_id'))
    is_face             = Column(Boolean)
    is_body             = Column(Boolean)
    face_x              = Column(DECIMAL(6, 3))
    face_y              = Column(DECIMAL(6, 3))
    face_z              = Column(DECIMAL(6, 3))
    mouth_gap           = Column(DECIMAL(6, 3))
    bbox                = Column(JSON)

class Clusters(Base):
    __tablename__ = 'Clusters'

    cluster_id     = Column(Integer, primary_key=True, autoincrement=True)
    cluster_median = Column(BLOB)

class ImagesClusters(Base):
    __tablename__ = 'ImagesClusters'

    image_id   = Column(Integer, ForeignKey('images.image_id'), primary_key=True)
    cluster_id = Column(Integer, ForeignKey('Clusters.cluster_id'))

class Poses(Base):
    __tablename__ = 'Poses'

    cluster_id     = Column(Integer, primary_key=True, autoincrement=True)
    cluster_median = Column(BLOB)

class ImagesPoses(Base):
    __tablename__ = 'ImagesPoses'

    image_id   = Column(Integer, ForeignKey('images.image_id'), primary_key=True)
    cluster_id = Column(Integer, ForeignKey('Poses.cluster_id'))

class Hands(Base):
    __tablename__ = 'Hands'

    cluster_id     = Column(Integer, primary_key=True, autoincrement=True)
    cluster_median = Column(BLOB)

class ImagesHands(Base):
    __tablename__ = 'ImagesHands'

    image_id   = Column(Integer, ForeignKey('images.image_id'), primary_key=True)
    cluster_id = Column(Integer, ForeignKey('Hands.cluster_id'))

class ClustersTemp(Base):
    __tablename__ = 'ClustersTemp'

    cluster_id     = Column(Integer, primary_key=True, autoincrement=True)
    cluster_median = Column(BLOB)

class ImagesClustersTemp(Base):
    __tablename__ = 'ImagesClustersTemp'

    image_id   = Column(Integer, ForeignKey('images.image_id'), primary_key=True)
    cluster_id = Column(Integer, ForeignKey('ClustersTemp.cluster_id'))

class BagOfKeywords(Base):
    __tablename__ = 'BagOfKeywords'
    image_id               = Column(Integer, primary_key=True, autoincrement=True)
    age_id                 = Column(Integer, ForeignKey('age.age_id'))
    gender_id              = Column(Integer, ForeignKey('gender.gender_id'))
    location_id            = Column(Integer, ForeignKey('location.location_id'))
    description            = Column(String(150))
    keyword_list           = Column(BLOB)  # Pickled list
    tokenized_keyword_list = Column(BLOB)  # Pickled list
    ethnicity_list         = Column(BLOB)  # Pickled list

class Topics(Base):
    __tablename__ = 'Topics' 
    topic_id = Column(Integer, primary_key=True, autoincrement=True)
    topic    = Column(String(150))
    
class ImagesTopics(Base):
    __tablename__ = 'ImagesTopics' 
    image_id    = Column(Integer, ForeignKey('images.image_id'), primary_key=True)
    topic_id    = Column(Integer, ForeignKey('Topics.topic_id'))
    topic_score = Column(Float)
    topic_id2    = Column(Integer, ForeignKey('Topics.topic_id'))
    topic_score2 = Column(Float)
    topic_id3    = Column(Integer, ForeignKey('Topics.topic_id'))
    topic_score3 = Column(Float)

class Topics_isnotface(Base):
    __tablename__ = 'Topics_isnotface' 
    topic_id = Column(Integer, primary_key=True, autoincrement=True)
    topic    = Column(String(150))
    
class ImagesTopics_isnotface(Base):
    __tablename__ = 'ImagesTopics_isnotface' 
    image_id    = Column(Integer, ForeignKey('images.image_id'), primary_key=True)
    topic_id    = Column(Integer, ForeignKey('Topics.topic_id'))
    topic_score = Column(Float)
    topic_id2    = Column(Integer, ForeignKey('Topics.topic_id'))
    topic_score2 = Column(Float)
    topic_id3    = Column(Integer, ForeignKey('Topics.topic_id'))
    topic_score3 = Column(Float)

class ImagesTopics_isnotface_isfacemodel(Base):
    __tablename__ = 'ImagesTopics_isnotface_isfacemodel' 
    image_id    = Column(Integer, ForeignKey('images.image_id'), primary_key=True)
    topic_id    = Column(Integer, ForeignKey('Topics.topic_id'))
    topic_score = Column(Float)
    topic_id2    = Column(Integer, ForeignKey('Topics.topic_id'))
    topic_score2 = Column(Float)
    topic_id3    = Column(Integer, ForeignKey('Topics.topic_id'))
    topic_score3 = Column(Float)
    
class imagestopics_ALLgetty4faces_isfacemodel(Base):
    __tablename__ = 'imagestopics_ALLgetty4faces_isfacemodel' 
    image_id    = Column(Integer, ForeignKey('images.image_id'), primary_key=True)
    topic_id    = Column(Integer, ForeignKey('Topics.topic_id'))
    topic_score = Column(Float)
    topic_id2    = Column(Integer, ForeignKey('Topics.topic_id'))
    topic_score2 = Column(Float)
    topic_id3    = Column(Integer, ForeignKey('Topics.topic_id'))
    topic_score3 = Column(Float)

class Topics48(Base):
    __tablename__ = 'Topics48' 
    topic_id = Column(Integer, primary_key=True, autoincrement=True)
    topic    = Column(String(150))
    
class ImagesTopics48(Base):
    __tablename__ = 'ImagesTopics48' 
    image_id    = Column(Integer, ForeignKey('images.image_id'), primary_key=True)
    topic_id    = Column(Integer, ForeignKey('Topics48.topic_id'))
    topic_score = Column(Float)
    topic_id2    = Column(Integer, ForeignKey('Topics48.topic_id'))
    topic_score2 = Column(Float)
    topic_id3    = Column(Integer, ForeignKey('Topics48.topic_id'))
    topic_score3 = Column(Float)

class ImagesBackground(Base):
    __tablename__ = 'ImagesBackground' 
    image_id     = Column(Integer, ForeignKey('images.image_id'), primary_key=True)
    hue          = Column(Float)
    lum          = Column(Float)
    sat          = Column(Float)
    val          = Column(Float)
    lum_torso    = Column(Float)
    hue_bb       = Column(Float)
    lum_bb       = Column(Float)
    sat_bb       = Column(Float)
    val_bb       = Column(Float)
    lum_torso_bb = Column(Float)
    selfie_bbox  = Column(JSON(none_as_null=True))
    is_left_shoulder= Column(Boolean)
    is_right_shoulder= Column(Boolean)

class SegmentTable(Base):
    __tablename__ = 'SegmentOct20'
    
    seg_image_id           = Column(Integer, primary_key=True, autoincrement=True)
    image_id = Column(Integer, primary_key=True)
    site_name_id           = Column(Integer, ForeignKey('site.site_name_id'))
    site_image_id          = Column(String(50))
    contentUrl             = Column(String(300), nullable=False)
    imagename              = Column(String(200))
    description            = Column(String(150))
    age_id                 = Column(Integer, ForeignKey('age.age_id'))
    gender_id              = Column(Integer, ForeignKey('gender.gender_id'))
    location_id            = Column(Integer, ForeignKey('location.location_id'))
    face_x                 = Column(DECIMAL(6, 3))
    face_y                 = Column(DECIMAL(6, 3))
    face_z                 = Column(DECIMAL(6, 3))
    mouth_gap              = Column(DECIMAL(6, 3))
    face_landmarks         = Column(BLOB)
    bbox                   = Column(JSON)
    face_encodings         = Column(BLOB)
    face_encodings68       = Column(BLOB)
    body_landmarks         = Column(BLOB)
    site_image_id          = Column(String(50), nullable=False)
    keyword_list           = Column(BLOB)  # Pickled list
    tokenized_keyword_list = Column(BLOB)  # Pickled list
    ethnicity_list         = Column(BLOB)  # Pickled list
    mongo_tokens           = Column(Boolean)
    mongo_body_landmarks   = Column(Boolean)
    mongo_face_landmarks   = Column(Boolean)
    mongo_body_landmarks_norm   = Column(Boolean)
    no_image               = Column(Boolean)
    two_noses              = Column(Boolean)
    is_dupe_of            = Column(Integer, ForeignKey('images.image_id'))
    mongo_hand_landmarks   = Column(Boolean)
    mongo_hand_landmarks_norm   = Column(Boolean)
    is_feet              = Column(Boolean)

    site     = relationship("Site")
    age      = relationship("Age")
    gender   = relationship("Gender")
    location = relationship("Location")

class SegmentBig(Base):
    __tablename__ = 'SegmentBig_isface'
    
    seg_image_id           = Column(Integer, primary_key=True, autoincrement=True)
    image_id               = Column(Integer, primary_key=True)
    site_name_id           = Column(Integer, ForeignKey('site.site_name_id'))
    site_image_id          = Column(String(50))
    contentUrl             = Column(String(300), nullable=False)
    imagename              = Column(String(200))
    description            = Column(String(150))
    age_id                 = Column(Integer, ForeignKey('age.age_id'))
    gender_id              = Column(Integer, ForeignKey('gender.gender_id'))
    location_id            = Column(Integer, ForeignKey('location.location_id'))
    face_x                 = Column(DECIMAL(6, 3))
    face_y                 = Column(DECIMAL(6, 3))
    face_z                 = Column(DECIMAL(6, 3))
    mouth_gap              = Column(DECIMAL(6, 3))
    face_landmarks         = Column(BLOB)
    bbox                   = Column(JSON)
    face_encodings68       = Column(BLOB)
    body_landmarks         = Column(BLOB)
    keyword_list           = Column(BLOB)  # Pickled list
    tokenized_keyword_list = Column(BLOB)  # Pickled list
    ethnicity_list         = Column(BLOB)  # Pickled list
    mongo_tokens           = Column(Boolean)

    site     = relationship("Site")
    age      = relationship("Age")
    gender   = relationship("Gender")
    location = relationship("Location")

class SegmentBig_isnotface(Base):
    __tablename__ = 'SegmentBig_ALLgetty4faces'
    
    seg_image_id           = Column(Integer, primary_key=True, autoincrement=True)
    image_id               = Column(Integer, primary_key=True)
    site_name_id           = Column(Integer, ForeignKey('site.site_name_id'))
    site_image_id          = Column(String(50))
    contentUrl             = Column(String(300), nullable=False)
    imagename              = Column(String(200))
    description            = Column(String(150))
    age_id                 = Column(Integer, ForeignKey('age.age_id'))
    gender_id              = Column(Integer, ForeignKey('gender.gender_id'))
    location_id            = Column(Integer, ForeignKey('location.location_id'))
    face_x                 = Column(DECIMAL(6, 3))
    face_y                 = Column(DECIMAL(6, 3))
    face_z                 = Column(DECIMAL(6, 3))
    mouth_gap              = Column(DECIMAL(6, 3))
    face_landmarks         = Column(BLOB)
    bbox                   = Column(JSON)
    face_encodings68       = Column(BLOB)
    body_landmarks         = Column(BLOB)
    keyword_list           = Column(BLOB)  # Pickled list
    tokenized_keyword_list = Column(BLOB)  # Pickled list
    ethnicity_list         = Column(BLOB)  # Pickled list
    mongo_tokens           = Column(Boolean)
    mongo_body_landmarks = Column(Boolean)
    mongo_face_landmarks = Column(Boolean)
    mongo_body_landmarks_norm = Column(Boolean)
    no_image = Column(Boolean)
    is_dupe_of = Column(Integer)
    mongo_hand_landmarks = Column(Boolean)
    mongo_hand_landmarks_norm = Column(Boolean)

    site     = relationship("Site")
    age      = relationship("Age")
    gender   = relationship("Gender")
    location = relationship("Location")


class PhoneBbox(Base):
    __tablename__ = 'PhoneBbox' 
    image_id = Column(Integer, ForeignKey('images.image_id'), primary_key=True)
    bbox_67     = Column(JSON(none_as_null=True))
    conf_67     = Column(Float)
    bbox_63     = Column(JSON(none_as_null=True))
    conf_63     = Column(Float)
    bbox_26     = Column(JSON(none_as_null=True))
    conf_26     = Column(Float)
    bbox_27     = Column(JSON(none_as_null=True))
    conf_27     = Column(Float)
    bbox_32     = Column(JSON(none_as_null=True))
    conf_32     = Column(Float)
    bbox_67_norm     = Column(JSON(none_as_null=True))
    bbox_63_norm     = Column(JSON(none_as_null=True))
    bbox_26_norm     = Column(JSON(none_as_null=True))
    bbox_27_norm     = Column(JSON(none_as_null=True))
    bbox_32_norm     = Column(JSON(none_as_null=True))

# these are for MM use for using segments
# class Clusters(Base):
#     __tablename__ = 'Clusters_May25segment123straight_lessrange'

#     cluster_id = Column(Integer, primary_key=True, autoincrement=True)
#     cluster_median = Column(BLOB)

# class ImagesClusters(Base):
#     __tablename__ = 'ImagesClusters_May25segment123straight_lessrange'

#     image_id = Column(Integer, ForeignKey('images.image_id'), primary_key=True)
#     cluster_id = Column(Integer, ForeignKey('Clusters_May25segment123straight_lessrange.cluster_id'))

class Counters(Base):
    __tablename__ = 'counters'
    counter_id = Column(Integer, primary_key=True, autoincrement=True)
    counter_name = Column(String(50))
    counter_value = Column(Integer);