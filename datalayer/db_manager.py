# -*- coding: utf-8 -*-

import yaml
import logging
import mysql.connector
from mysql.connector import errorcode

from datalayer.node.winmodule_hash_node import WinModuleHashNode
from datalayer.hash_algorithm.hash_algorithm import HashAlgorithm
from datalayer.hash_algorithm.tlsh_algorithm import TLSHHashAlgorithm
from datalayer.hash_algorithm.ssdeep_algorithm import SSDEEPHashAlgorithm
from datalayer.database.operating_system import OS
from datalayer.database.module import Module

from common.errors import HashValueNotInDBError, PageIdValueNotInDBError

SQL_GET_ALL_PAGES = """
    SELECT p.{}, m.id AS module_id, m.file_version, m.original_filename, 
        m.internal_filename, m.product_filename, m.company_name, 
        m.legal_copyright, m.classification, m.size, m.base_address, o.*
    FROM pages p
    JOIN modules m ON p.module_id = m.id 
    JOIN os o ON m.os_id = o.id
    """

SQL_GET_WINMODULE_BY_HASH = """
    SELECT p.{}, m.id AS module_id, m.file_version, m.original_filename, 
        m.internal_filename, m.product_filename, m.company_name, 
        m.legal_copyright, m.classification, m.size, m.base_address, o.*
    FROM pages p
    JOIN modules m ON p.module_id = m.id
    JOIN os o ON m.os_id = o.id
    WHERE p.{} = %s
"""

SQL_GET_WINMODULE_BY_PAGEID = """
    SELECT p.{}, m.id AS module_id, m.file_version, m.original_filename, 
        m.internal_filename, m.product_filename, m.company_name, 
        m.legal_copyright, m.classification, m.size, m.base_address, o.*
    FROM pages p
    JOIN modules m ON p.module_id = m.id
    JOIN os o ON m.os_id = o.id
    WHERE p.id = %s
"""


logger = logging.getLogger(__name__)

class DBManager():
    def __init__(self):
        self.load_credentials()
        self.connect()
    
    def load_credentials(self):
        with open('settings.yaml', 'r') as file:
            self.config = yaml.safe_load(file)

    def connect(self):
        try:
            self.connection = mysql.connector.connect(
                host=self.config["db_host"],
                user=self.config["db_user"],
                password=self.config["db_password"],
                database=self.config["db_name"],
                port=self.config["db_port"]
            )
            self.cursor = self.connection.cursor(dictionary=True)
        except mysql.connector.Error as err:
            if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
                logger.error("Invalid database credentials")
            elif err.errno == errorcode.ER_BAD_DB_ERROR:
                logger.error("Database does not exist")
            else:
                logger.error(err)
            raise

    def _clean_dict_keys(self, _dict: dict, keys: list):
        for key in keys:
            _dict.pop(key, None)

    def _row_to_module(self, row):
        return  Module(
            os=OS(row["id"], row["name"], row["version"]),
            id=row["module_id"],
            file_version=row["file_version"],
            original_filename=row["original_filename"],
            internal_filename=row["internal_filename"],
            product_name=row["product_filename"],
            company_name=row["company_name"],
            legal_copyright=row["legal_copyright"],
            classification=row["classification"],
            size=row["size"],
            base_address=row["base_address"]
            )


    def get_winmodule_data_by_pageid(self, page_id=0, algorithm=HashAlgorithm):
        logger.info(f"Getting results for \"{page_id}\" from DB ({algorithm.__name__}) ...")
        hash_column = "hashTLSH" if algorithm == TLSHHashAlgorithm else "hashSSDEEP"
        query = SQL_GET_WINMODULE_BY_PAGEID.format(hash_column)
        self.cursor.execute(query, (page_id,))
        row = self.cursor.fetchone()
        if not row:
            logger.debug(f"Error! Page ID {page_id} not in DB (algorithm: {algorithm})")
            raise PageIdValueNotInDBError
        
        module = self._row_to_module(row)
        return WinModuleHashNode(id=row[hash_column], hash_algorithm=algorithm, module=module)


    def get_winmodule_data_by_hash(self, algorithm: str = "", hash_value: str = ""):
        logger.info(f"Getting results for \"{hash_value}\" from DB ({algorithm})")
        hash_column = "hashTLSH" if algorithm == TLSHHashAlgorithm else "hashSSDEEP"
        query = SQL_GET_WINMODULE_BY_HASH.format(hash_column, hash_column)
        self.cursor.execute(query, (hash_value,))
        row = self.cursor.fetchone()
        if not row:
            logger.debug(f"Error! Hash value {hash_value} not in DB (algorithm: {algorithm})")
            raise HashValueNotInDBError
        
        module = self._row_to_module(row)
        return WinModuleHashNode(id=hash_value, hash_algorithm=algorithm, module=module)


    def get_winmodules(self, algorithm: HashAlgorithm = TLSHHashAlgorithm, limit: int = None, modules_of_interest: set = None) -> set:
        try:
            winmodules = set()
            operating_systems = set()
            modules = set()

            hash_column = "hashTLSH" if algorithm == TLSHHashAlgorithm else "hashSSDEEP"
            query = SQL_GET_ALL_PAGES.format(hash_column)
            if limit:
                query = query + f" LIMIT {limit}"
            self.cursor.execute(query)
            results = self.cursor.fetchall()

            for row in results:
                os_id = row["id"]
                os_version = row["version"]
                os_name = row["name"]
                hash_value = row[hash_column]

                current_os = OS(os_id, os_name, os_version)
                if current_os in operating_systems:
                    current_os = next(os for os in operating_systems if os == current_os)
                else:
                    operating_systems.add(current_os)

                current_module = self._row_to_module(row)

                # Supposedly more memory-efficient, but it slows down retrieval
                '''
                if current_module in modules:
                    current_module = next(module for module in modules if module == current_module)
                else:
                    modules.add(current_module)
                '''

                if modules_of_interest and current_module.internal_filename not in modules_of_interest:
                    continue
                
                winmodules.add(WinModuleHashNode(hash_value, algorithm, current_module))

            return winmodules
        except mysql.connector.Error as err:
            logger.error(f"Database query error: {err}")
            raise
        finally:
            self.cursor.close()
        
    def close(self):
        self.cursor.close()
        self.connection.close()