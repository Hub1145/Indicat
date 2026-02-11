import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, String, Integer, DateTime, JSON, Float, ForeignKey, Boolean
from datetime import datetime
import uuid

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://user:password@db:5432/tadb")

engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    api_key = Column(String, unique=True, index=True)
    tier = Column(String, default="free") # free, pro, enterprise
    created_at = Column(DateTime, default=datetime.utcnow)
    monthly_requests = Column(Integer, default=0)

class APIRequest(Base):
    __tablename__ = "api_requests"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, ForeignKey("users.id"))
    endpoint = Column(String)
    symbols = Column(JSON)
    indicators = Column(JSON)
    response_time_ms = Column(Integer)
    status_code = Column(Integer)
    timestamp = Column(DateTime, default=datetime.utcnow)

class Alert(Base):
    __tablename__ = "alerts"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"))
    symbol = Column(String)
    condition = Column(JSON) # e.g., {"indicator": "RSI", "operator": "<", "value": 30}
    webhook_url = Column(String)
    active = Column(Boolean, default=True)
    last_triggered = Column(DateTime, nullable=True)

async def init_db():
    async with engine.begin() as conn:
        # In production, use Alembic. For this task, we can create tables here.
        await conn.run_sync(Base.metadata.create_all)

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
