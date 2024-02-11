from sqlalchemy import Column, Integer, String, BigInteger, ForeignKey
from sqlalchemy.orm import declarative_base
from .base import Base
import logging
logger = logging.getLogger(__name__)

class Page(Base):
    __tablename__ = 'pages'

    id = Column(BigInteger, primary_key=True)
    num_page = Column(Integer)
    preprocess_method = Column(String)
    hashTLSH = Column(String)
    hashSSDEEP = Column(String)
    #FIXME this column name should be hashSDHASH
    hashSD = Column(String)
    module_id = Column(BigInteger, ForeignKey('modules.id'))
    
    def as_dict(self):
       return {c.name: getattr(self, c.name) for c in self.__table__.columns} 

    def __str__(self):
        return str(self.as_dict())

    def __repr__(self):
        return str(self.as_dict())

    # other_page should be a Page
    def _safe_compare(self, other_page):
        # check TLSH first, then SSDEEP, and finally SDHASH
        equal_size = True
        if len(self.hashTLSH) != len(other_page.hashTLSH):
            equal_size = False # size
        equal_TLSH = -1
        if equal_size:
            for i in range(0, len(other_page.hashTLSH), 1): # unnecesary checking, fixed length
                equal_TLSH = ord(self.hashTLSH[i]) - ord(other_page.hashTLSH[i])
        aux = '' if self.id != other_page.id and equal_TLSH == 0 and equal_size else 'NOT '
        logger.debug(f"[*]{self.id} != {other_page.id} -> TLSH {aux}MATCH")
        
        if len(self.hashSSDEEP) != len(other_page.hashTLSH):
            equal_size = False # size
        equal_SSDEEP = -1 
        if equal_size:
            for i in range(0, len(other_page.hashSSDEEP), 1):
                equal_SSDEEP = (ord(self.hashSSDEEP[i]) - ord(other_page.hashSSDEEP[i]))
        aux = '' if self.id != other_page.id and equal_SSDEEP == 0 and equal_size else 'NOT '
        logger.debug(f"[*]{self.id} != {other_page.id} -> SSDEEP {aux}MATCH")
        
        equal_SDHASH = -1 
        if len(self.hashSD) != len(other_page.hashSD):
            equal_size = False # distinct size
        if equal_size:
            for i in range(0, len(other_page.hashSD), 1):
                equal_SSDHASH = (ord(self.hashSD[i]) - ord(other_page.hashSD[i]))
        aux = '' if self.id != other_page.id and equal_SDHASH == 0 and equal_size else 'NOT '
        logger.debug(f"[*]{self.id} != {other_page.id} -> SDHASH {aux}MATCH")
        return (equal_TLSH == 0), (equal_SSDEEP == 0), (equal_SDHASH == 0)

    # other_page should be Page
    def is_equal(self, other_page):
        col1, col2, col3 =  self._safe_compare(other_page)
        if col1 and (not col2 or not col3):
            logger.critical(f"[-] TLSH COLLISION [#page {self.id}:{other_page.id}]\n\"{self.hashTLSH}\" and \"{other_page.hashTLSH}\"")
        if col2 and (not col1 or not col3):
            logger.critical(f"[-] SSDEEP COLLISION [#page {self.id}:{other_page.id}]\n\"{self.hashSSDEEP}\" and \"{other_page.hashSSDEEP}\"")
        if col3 and (not col1 or not col2):
            logger.critical(f"[-] SDHASH COLLISION [#page {self.id}:{other_page.id}]\n\"{self.hashSD}\" and \"{other_page.hashSD}\"")
        
        # three hashes must be equal to assure both are equal -- this means the content is the same (other conditions above)
        return col1 and col2 and col3
