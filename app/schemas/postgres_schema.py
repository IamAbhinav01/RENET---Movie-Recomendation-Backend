from app.config.db_Config import engine
from sqlalchemy import Integer,String,Column,Text,Float,ForeignKey
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base


Session = sessionmaker(bind = engine)
session = Session()


Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)

class Item(Base):
    __tablename__ = 'items'
    id = Column(Integer, primary_key=True)
    title = Column(Text,nullable=False)
    genres = Column(Text,nullable=False)
    primary_genre = Column(Text,nullable=False)
    poster_url = Column(Text, nullable=True)
    plot = Column(Text, nullable=True)


class Interactions(Base):
    __tablename__ = 'interactions'
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    item_id = Column(Integer, ForeignKey('items.id'), nullable=False)
    rating = Column(Float, nullable=False)
    event_type = Column(String, nullable=False, default='rating')
