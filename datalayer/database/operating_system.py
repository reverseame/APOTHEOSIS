from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy import create_engine, Column, Integer, String, BigInteger
from .base import Base

class OS(Base):
    __tablename__ = 'os'

    id = Column(BigInteger, primary_key=True)
    name = Column(String)
    version = Column(String)
    modules = relationship("Module")

    def __str__(self):
        return f"Name: {self.name}, Version: {self.version}"
    
    def as_dict(self):
       return {str('os_' + c.name): getattr(self, c.name) for c in self.__table__.columns}
