# -*- coding: utf-8 -*-

#TODO docstring
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
import yaml
import logging
logger = logging.getLogger(__name__)

import datalayer.database.base
from datalayer.database.operating_system import OS
from datalayer.database.module import Module
from datalayer.database.page import Page

from datalayer.node.winmodule_hash_node import WinModuleHashNode
from datalayer.hash_algorithm.tlsh_algorithm import TLSHHashAlgorithm
from datalayer.hash_algorithm.ssdeep_algorithm import SSDEEPHashAlgorithm

from datalayer.errors import HashValueNotInDBError

class DBManager():
    
    def __init__(self):
        self.connect()

    def load_credentials(self):
        with open('settings.yaml', 'r') as file:
            self.config = yaml.safe_load(file)

    def connect(self):
        self.load_credentials()
        database = self.config["db_name"]
        host = self.config["db_host"]
        user = self.config["db_user"]
        password = self.config["db_password"]
        port = self.config["db_port"]
        self.engine = create_engine(f'mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}')
        self.session = sessionmaker(bind=self.engine)()

    def _clean_dict_keys(self, _dict: dict, keys: list):
        """
        Raises KeyError if key is not found in _dict
        """
        for key in keys:
            del _dict[key]

    def get_winmodule(self, algorithm, hash_value):
        logger.info(f"Getting results for \"{hash_value}\" from DB ({algorithm})")
        # construct statement for to retrieve winmodule associated to the given hash value
        stmt = select(Page, Module).filter(
                    Page.hashTLSH == hash_value if algorithm == "tlsh" else Page.hashSSDEEP == hash_value
                    ).filter(
                    Page.module_id == Module.id
                    )
        # it should be only one result (one Page and one Module)
        row = self.session.execute(stmt).first()
        if row is None: # hash value is NOT in database
            logger.debug(f"Error! Hash value {hash_value} not in DB (algorithm: {algorithm})")
            raise HashValueNotInDBError

        # merge results of both tables in a single dict
        results = {}
        for i in range(0, len(row)):
            results.update(row[i].as_dict())            

        # clean unnecesary keys in results
        keys_to_remove = ['id', 'module_id', 'preprocess_method', 'os_id', 'hashTLSH', 'hashSD', 'hashSSDEEP']
        logger.debug(f"Cleaning result keys {keys_to_remove} ...")
        self._clean_dict_keys(results, keys_to_remove)
        
        return results

    def get_winmodules(self, algorithm, limit: int = None):
        winmodules = []

        #TODO limit this query and remove control lines for limit in the loop below
        query = self.session.query(OS).all()

        for os in query:
            for module in os.modules:
                for page in module.pages:
                    # limit results
                    if limit is not None and len(winmodules) >= limit:
                        return winmodules
                    # avoid non-computed hashes in the results
                    if algorithm == TLSHHashAlgorithm and page.hashTLSH != "-":
                        winmodules.append(WinModuleHashNode(page.hashTLSH, TLSHHashAlgorithm, module))
                    elif algorithm == SSDEEPHashAlgorithm and page.hashSSDEEP != "-":
                        winmodules.append(WinModuleHashNode(page.hashSSDEEP, SSDEEPHashAlgorithm, module))

        return winmodules

    def close(self):
        self.session.close()

