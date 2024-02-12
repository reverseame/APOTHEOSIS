from sqlalchemy import Column, Integer, String, BigInteger, ForeignKey
from sqlalchemy.orm import declarative_base
from .base import Base

class Page(Base):
    __tablename__ = 'pages'

    id = Column(BigInteger, primary_key=True)
    num_page = Column(Integer)
    preprocess_method = Column(String)
    hashTLSH = Column(String)
    hashSSDEEP = Column(String)
    hashSD = Column(String)
    module_id = Column(BigInteger, ForeignKey('modules.id'))
    
    def as_dict(self):
       return {c.name: getattr(self, c.name) for c in self.__table__.columns} 

    def __str__(self):
        return str(self.as_dict())

    def __repr__(self):
        return str(self.as_dict())

