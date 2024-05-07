from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy_utils import EncryptedType
from cryptography.fernet import Fernet

Base = declarative_base()

# Generate key for encryption
key = Fernet.generate_key()
fernet = Fernet(key)

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True, index=True)
    full_name = Column(String)
    hashed_password = Column(EncryptedType(String, fernet))

# Database URL should be configured in your environment
engine = create_engine('DATABASE_URL', echo=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base.metadata.create_all(bind=engine)
