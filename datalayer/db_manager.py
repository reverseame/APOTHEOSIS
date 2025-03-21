# -*- coding: utf-8 -*-

import yaml
import logging
import mysql.connector
from mysql.connector import errorcode

from datalayer.hash_algorithm.hash_algorithm import HashAlgorithm
from datalayer.hash_algorithm.tlsh_algorithm import TLSHHashAlgorithm
from datalayer.hash_algorithm.ssdeep_algorithm import SSDEEPHashAlgorithm
from datalayer.database.operating_system import OS
from datalayer.database.module import Module

from common.errors import HashValueNotInDBError, PageIdValueNotInDBError

SQL_GET_ALL_OS = """
    SELECT *
    FROM os
    """
SQL_GET_MODULES_BY_OS = """
    SELECT m.id AS module_id, m.file_version, m.original_filename, 
        m.internal_filename, m.product_filename, m.company_name, 
        m.legal_copyright, m.classification, m.size, m.base_address
    FROM modules m
    WHERE m.os_id = %s
    """

SQL_GET_ALL_PAGES = """
    SELECT p.{}, m.id AS module_id, m.file_version, m.original_filename, 
        m.internal_filename, m.product_filename, m.company_name, 
        m.legal_copyright, m.classification, m.size, m.base_address, o.*
    FROM pages p
    JOIN modules m ON p.module_id = m.id 
    JOIN os o ON m.os_id = o.id
    """

SQL_GET_ALL_PAGES_LAZY = """
    SELECT p.{}
    FROM pages p
    JOIN modules m ON p.module_id = m.id
    JOIN os o ON m.os_id = o.id
    """

SQL_GET_MODULE_BY_HASH = """
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

    def _row_to_module(self, row, os=None):
        if not os:
            os = OS(row["id"], row["name"], row["version"])
        return  Module(
            os=os,
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

    def get_winpage_module_by_hash(self, algorithm : HashAlgorithm, hash_value: str = ""):
        logger.info(f"Getting results for \"{hash_value}\" from DB ({algorithm})")
        hash_column = "hashTLSH" if algorithm == TLSHHashAlgorithm else "hashSSDEEP"

        query = SQL_GET_MODULE_BY_HASH.format(hash_column, hash_column)
        self.cursor.execute(query, (hash_value,))
        row = self.cursor.fetchone()

        if not row:
            logger.debug(f"Error! Hash value {hash_value} not in DB (algorithm: {algorithm})")
            raise HashValueNotInDBError
        
        module = self._row_to_module(row)
        return module
    
    def get_organized_modules(self, algorithm: HashAlgorithm = TLSHHashAlgorithm) -> dict:
        result = {}

        self.cursor.execute(SQL_GET_ALL_OS)
        db_operating_systems = self.cursor.fetchall()
        for db_os in db_operating_systems:
            os_name = db_os['version']
            result[os_name] = {}

            self.cursor.execute(SQL_GET_MODULES_BY_OS, (db_os['id'],))
            modules = self.cursor.fetchall()
            for module in modules:
                module_name = module['internal_filename']
                original_name = module['original_filename']
                result[os_name][original_name] = set()

                pages = self.get_winmodules(algorithm, modules_of_interest={module_name}, os_id=db_os['id'])
                for page in pages:
                    result[os_name][original_name].add(page)

        return result
    
    def get_winmodules(self, algorithm: HashAlgorithm = TLSHHashAlgorithm, limit: int = None, modules_of_interest: set = None, os_id: int = None,
                       lazy: bool = True) -> set:
        from datalayer.node.winpage_hash_node import WinPageHashNode # Avoid circular deps   
        try:
            query = SQL_GET_ALL_PAGES_LAZY if lazy else SQL_GET_ALL_PAGES

            winmodules = set()
            

            hash_column = "hashTLSH" if algorithm == TLSHHashAlgorithm else "hashSSDEEP"
            query = SQL_GET_ALL_PAGES.format(hash_column)  # Inject hash column

            conditions = []
            params = []

            if modules_of_interest:
                placeholders = ', '.join(['%s'] * len(modules_of_interest))
                conditions.append(f"m.internal_filename IN ({placeholders})")
                params.extend(modules_of_interest)

            if os_id is not None:
                conditions.append("o.id = %s")
                params.append(os_id)

            if algorithm == TLSHHashAlgorithm:
                conditions.append("p.hashTLSH != '*'")
                conditions.append("p.hashTLSH != '-'")

            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            if limit:
                query += " LIMIT %s"
                params.append(limit)

            self.cursor.execute(query, params)
            results = self.cursor.fetchall()

            if lazy:
                for row in results:
                    hash_value = row[hash_column]
                    winmodules.add(WinPageHashNode(hash_value, algorithm, None))
            else:
                operating_systems = set()
                modules = set()
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
                    winmodules.add(WinPageHashNode(hash_value, algorithm, current_module))

            return winmodules
        except mysql.connector.Error as err:
            logger.error(f"Database query error: {err}")
            raise
        finally:
            pass
            #self.cursor.close()
        
    def close(self):
        self.cursor.close()
        self.connection.close()