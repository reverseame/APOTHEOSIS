from sqlalchemy import Column, Integer, String, BigInteger, ForeignKey
from sqlalchemy.orm import relationship, declarative_base
from base import Base

class Module(Base):
    __tablename__ = 'modules'

    id = Column(BigInteger, primary_key=True)
    file_version = Column(String)
    original_filename = Column(String)
    internal_filename = Column(String)
    product_filename = Column(String)
    company_name = Column(String)
    legal_copyright = Column(String)
    classification = Column(String)
    size = Column(Integer)
    base_address = Column(BigInteger)
    os_id = Column(BigInteger, ForeignKey('os.id'))

    pages = relationship("Page")
    os = relationship("OS", back_populates="modules")

    def as_dict(self):
       return {c.name: str(getattr(self, c.name)) for c in self.__table__.columns} | self.os.as_dict()


