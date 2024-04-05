from sqlalchemy import Column, Integer, String, BigInteger, ForeignKey
from sqlalchemy.orm import declarative_base
from .base import Base
import logging
logger = logging.getLogger(__name__)

class Page(Base):
    __tablename__ = 'pages'

    # April 05, 2024: Updated for dataset DB
    id                = Column(BigInteger, primary_key=True)
    num_page          = Column(Integer)
    preprocess_method = Column(String)
    hashTLSH          = Column(String)
    hashSSDEEP        = Column(String)
    hashSDHASH        = Column(String)
    module_id         = Column(BigInteger, ForeignKey('modules.id'))
    
    def as_dict(self):
       return {c.name: getattr(self, c.name) for c in self.__table__.columns} 

    def __str__(self):
        return str(self.as_dict())

    def __repr__(self):
        return str(self.as_dict())

    # other_page should be a Page
    def _compare(self, other_page):
        equal_TLSH = self.hashTLSH == other_page.hashTLSH
        equal_SSDEEP = self.hashSSDEEP == other_page.hashSSDEEP
        equal_SDHASH = self.hashSDHASH == other_page.hashSDHASH

        return equal_TLSH, equal_SSDEEP, equal_SDHASH

    # other_page should be Page
    def is_equal(self, other_page) -> (bool, list):
        """Returns a bool flag and a 3-bool list indicating if the page is equal to other page (all hashes match).
        The 3-bool list contains the result of comparing TLSH, SSDEEP, and SDHASH hashes. The flag return
        value is simply an "and" of the elements in the 3-bool list.

        Arguments:
        other_page  -- Page to compare with
        """
        col1, col2, col3 =  self._compare(other_page)
        if col1 and (not col2 or not col3):
            logger.critical(f"[-] TLSH COLLISION [#pages {self.id}:{other_page.id}]\n\"{self.hashTLSH}\" and \"{other_page.hashTLSH}\"")
        if col2 and (not col1 or not col3):
            logger.critical(f"[-] SSDEEP COLLISION [#pages {self.id}:{other_page.id}]\n\"{self.hashSSDEEP}\" and \"{other_page.hashSSDEEP}\"")
        if col3 and (not col1 or not col2):
            logger.critical(f"[-] SDHASH COLLISION [#pages {self.id}:{other_page.id}]\n\"{self.hashSDHASH}\" and \"{other_page.hashSDHASH}\"")
        
        # three hashes must be equal to assure both are equal -- this means the content is the same (other conditions above)
        return (col1 and col2 and col3), [col1, col2, col3]
