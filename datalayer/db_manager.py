# -*- coding: utf-8 -*-

#TODO docstring
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker, scoped_session, joinedload
import yaml
import logging
logger = logging.getLogger(__name__)

import datalayer.database.base
from datalayer.database.operating_system import OS
from datalayer.database.module import Module
from datalayer.database.page import Page

from datalayer.node.winmodule_hash_node import WinModuleHashNode
from datalayer.hash_algorithm.hash_algorithm import HashAlgorithm
from datalayer.hash_algorithm.tlsh_algorithm import TLSHHashAlgorithm
from datalayer.hash_algorithm.ssdeep_algorithm import SSDEEPHashAlgorithm

from common.errors import HashValueNotInDBError
from common.errors import PageIdValueNotInDBError

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
        self.engine = create_engine(f'mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}', pool_recycle=7200)
        # create scoped sessions to retrieve data, see below
        # https://stackoverflow.com/questions/26891971/mysql-connection-not-available-when-use-sqlalchemymysql-and-flask

    def _clean_dict_keys(self, _dict: dict, keys: list):
        """
        Raises KeyError if key is not found in _dict
        """
        for key in keys:
            del _dict[key]

    def get_winmodule_data_by_pageid(self, page_id=0, algorithm=HashAlgorithm):
        logger.info(f"Getting results for \"{page_id}\" from DB ({algorithm.__name__}) ...")
        # construct statement for to retrieve winmodule associated to the given page id
        session = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=self.engine))
        stmt = select(Page, Module).filter(
                    Page.id == page_id
                    ).filter(
                    Page.module_id == Module.id
                    )
        # it should be only one result (one Page and one Module)
        row = session.execute(stmt).first()
        if row is None: # hash value is NOT in database
            logger.debug(f"Error! value {page_id} not in DB")
            raise PageIdValueNotInDBError

        page    = row[0]
        module  = row[1]
        # XXX this may need a more complex logic if we use more hashes
        hash_value = page.hashTLSH if algorithm == TLSHHashAlgorithm else page.hashSSDEEP
        # create the node now 
        win_module_hash_node = WinModuleHashNode(page.hashTLSH, TLSHHashAlgorithm, module=module, page=page)
        
        session.close()
        return win_module_hash_node

    def get_winmodule_data_by_hash(self, algorithm: str="", hash_value: str=""):
        logger.info(f"Getting results for \"{hash_value}\" from DB ({algorithm})")
        session = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=self.engine))

        # construct statement for to retrieve winmodule associated to the given hash value
        stmt = select(Page, Module).filter(
                    Page.hashTLSH == hash_value if algorithm == "tlsh" else Page.hashSSDEEP == hash_value
                    ).filter(
                    Page.module_id == Module.id
                    )
        # it should be only one result (one Page and one Module)
        row = session.execute(stmt).first()
        if row is None: # hash value is NOT in database
            logger.debug(f"Error! Hash value {hash_value} not in DB (algorithm: {algorithm.__name__})")
            raise HashValueNotInDBError

        # merge results of both tables in a single dict
        results = {}
        for i in range(0, len(row)):
            results.update(row[i].as_dict())            

        # clean unnecesary keys in results
        keys_to_remove = ['id', 'module_id', 'preprocess_method', 'os_id', 'hashTLSH', 'hashSDHASH', 'hashSSDEEP']
        logger.debug(f"Cleaning keys {keys_to_remove} in the result ...")
        self._clean_dict_keys(results, keys_to_remove)
        session.close()

        return results

    def get_winmodules(self, algorithm, limit: int = None, modules_of_interest: list=None) -> (list, list):
        """Return subset of pages of modules of interest (list of WinModuleHashNode).
        """
        modules_dict = {}
        winmodules = []

        session = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=self.engine, expire_on_commit=False))
        query = session.query(OS).options(joinedload(OS.modules).joinedload(Module.pages)).all()
        for os in query:
            for module in os.modules:
                # check if this module is of interest
                module_name = module.internal_filename.replace('.dll', '') # remove 'dll' from internal_filename, if exists
                if modules_of_interest and (module_name not in modules_of_interest):
                    continue
                # add it to the modules dict
                if modules_dict.get(module.id) is None:
                    modules_dict[module.id] = module

                module_ptr = modules_dict[module.id]
                for page in module.pages:
                    # limit results
                    if limit is not None and len(winmodules) >= limit:
                        return winmodules, modules_dict
                    # avoid non-computed hashes in the results
                    if algorithm == TLSHHashAlgorithm and page.hashTLSH != "-":
                        winmodules.append(WinModuleHashNode(page.hashTLSH, TLSHHashAlgorithm, module=module, page=page ))
                    elif algorithm == SSDEEPHashAlgorithm and page.hashSSDEEP != "-":
                        winmodules.append(WinModuleHashNode(page.hashSSDEEP, SSDEEPHashAlgorithm, module=module, page=page))

        session.close()
        return winmodules, modules_dict
