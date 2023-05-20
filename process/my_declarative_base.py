from sqlalchemy import Column, Integer, String, Date, Boolean, DECIMAL, BLOB, ForeignKey, JSON
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

class Site(Base):
    __tablename__ = 'site'
    site_name_id = Column(Integer, primary_key=True, autoincrement=True)
    site_name = Column(String(20))

class Location(Base):
    __tablename__ = 'location'
    location_id = Column(Integer, primary_key=True, autoincrement=True)
    location_text = Column(String(50))
    location_number = Column(String(50))
    location_code = Column(String(50))

class Images(Base):
    __tablename__ = 'images'
    image_id = Column(Integer, primary_key=True, autoincrement=True)
    site_name_id = Column(Integer, ForeignKey('site.site_name_id'))
    site_image_id = Column(String(50), nullable=False)
    age_id = Column(Integer, ForeignKey('age.age_id'))
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
    face_encodings = Column
