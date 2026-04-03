from sqlalchemy import Column, Integer, String, Float, JSON, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import datetime
import os

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./autopilot.db")

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    clerk_id = Column(String, unique=True, index=True)
    email = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    files = relationship("File", back_populates="owner")
    experiments = relationship("Experiment", back_populates="owner")

class File(Base):
    __tablename__ = "files"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    filename = Column(String)
    original_name = Column(String)
    filepath = Column(String)
    upload_date = Column(DateTime, default=datetime.datetime.utcnow)
    
    owner = relationship("User", back_populates="files")

class Experiment(Base):
    __tablename__ = "experiments"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    model_name = Column(String)
    target_column = Column(String)
    metrics = Column(JSON)
    model_path = Column(String)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    owner = relationship("User", back_populates="experiments")

# Engine and sessionmaker are now lazy-loaded to prevent hangs at import time
_engine = None
_SessionLocal = None

def get_engine():
    global _engine
    if _engine is None:
        # Only use SQLite-specific connect_args if the URL starts with sqlite
        is_sqlite = DATABASE_URL.startswith("sqlite")
        engine_args = {"connect_args": {"check_same_thread": False}} if is_sqlite else {}
        _engine = create_engine(DATABASE_URL, **engine_args)
    return _engine

def get_session_local():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=get_engine())
    return _SessionLocal

def get_db():
    SessionLocal = get_session_local()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    Base.metadata.create_all(bind=get_engine())
