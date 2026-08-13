from config.db_Config import engine
from sqlalchemy import Integer,String,Column,Text
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


SQL_SCHEMA = """
DROP TABLE IF EXISTS interactions CASCADE;
DROP TABLE IF EXISTS items CASCADE;
DROP TABLE IF EXISTS users CASCADE;

CREATE TABLE users (
    id INTEGER PRIMARY KEY
);

CREATE TABLE items (
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    genres TEXT NOT NULL,
    primary_genre TEXT NOT NULL
);

CREATE TABLE interactions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    item_id INTEGER NOT NULL REFERENCES items(id),
    rating REAL NOT NULL,
    event_type TEXT NOT NULL DEFAULT 'rating',
    created_at TIMESTAMP NOT NULL DEFAULT now()
);

CREATE INDEX idx_interactions_user ON interactions(user_id);
CREATE INDEX idx_interactions_item ON interactions(item_id);

"""