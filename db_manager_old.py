import mysql.connector
from operating_system import OperatingSystem
from module import Module
from page import Page
from node_winmodules import WinmodulesHashNode
import yaml
import logging
import json

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.getLogger('mysql.connector').setLevel(logging.WARNING)
logging.getLogger('psycopg2').setLevel(logging.WARNING)

SQL_GET_OS = "SELECT * FROM os"
SQL_GET_MODULES = "SELECT * FROM module WHERE version = %s and name = %s"
SQL_GET_PAGES = "SELECT * FROM pages"
SQL_GET_PAGES_LIMIT = "SELECT * FROM pages LIMIT %s"
SQL_GET_MODULE_FROM_HASH = "SELECT DISTINCT ( \
            'file_version', m.file_version, \
            'original_filename', m.original_filename, \
            'internal_filename', m.internal_filename, \
            'product_filename', m.product_filename, \
            'company_name', m.company_name, \
            'legal_copyright', m.legal_copyright, \
            'classification', m.classification, \
            'size', m.size, \
            'base_address', m.base_address, \
            'os_name', o.name, \
            'os_version', o.version \
        ) \
        FROM module m \
        INNER JOIN os o ON m.os_id = o.id \
        WHERE m.id = (SELECT module_id FROM pages WHERE %s = %s);"
SQL_GET_MODULE_FROM_HASH_JSON = "SELECT DISTINCT JSON_OBJECT( \
            'file_version', m.file_version, \
            'original_filename', m.original_filename, \
            'internal_filename', m.internal_filename, \
            'product_filename', m.product_filename, \
            'company_name', m.company_name, \
            'legal_copyright', m.legal_copyright, \
            'classification', m.classification, \
            'size', m.size, \
            'base_address', m.base_address, \
            'os_name', o.name, \
            'os_version', o.version \
        ) AS json_data \
        FROM module m \
        INNER JOIN os o ON m.os_id = o.id \
        WHERE m.id = (SELECT module_id FROM pages WHERE hashTLSH = %s);"


class DBManager:
    def __init__(self):
        self.conn = None

    def connect(self):
        self.load_credentials()
        self.conn = mysql.connector.connect(database = self.config["db_name"],
            host = self.config["db_host"],
            user = self.config["db_user"],
            password = self.config["db_password"],
            port = self.config["db_port"])
        
    def load_credentials(self):
        with open('settings.yaml', 'r') as file:
            self.config = yaml.safe_load(file)

    def disconnect(self):
        if self.conn:
            self.conn.close()
            self.conn = None

    def get_os_data(self):
        self.connect()
        cursor = self.conn.cursor()
        cursor.execute(SQL_GET_OS)
        data = cursor.fetchall()
        os_list = []
        for row in data:
            os_list.append(OperatingSystem(*row))
        self.disconnect()
        return os_list
    
    def get_modules(self, os):
        self.connect()
        cursor = self.conn.cursor()
        modules = []
        try: 
            cursor.execute(SQL_GET_MODULES, (os.version, os.name))
            data = cursor.fetchall()
            for row in data:
                modules.append(Module(*row))
            self.disconnect()
            return modules
        except mysql.connector.errors.ProgrammingError as Error:
            print(f"Error: {Error}")
            return []
    
    def get_pages(self, module):
        pages = []
        try:
            self.connect()
            cursor = self.conn.cursor()
            cursor.execute(SQL_GET_PAGES, (module.fileVersion, module.originalFilename, module.internalFilename))
            data = cursor.fetchall()
            
            for row in data:
                page = Page(*row)
                if page.hashTLSH != "-":
                    pages.append(Page(*row))
            self.disconnect()
        except mysql.connector.errors.ProgrammingError as error:
            logger.warning(error)

        return pages

    
    def get_all_pages(self):
        pages = []
        try:
            self.connect()
            cursor = self.conn.cursor()
            cursor.execute(SQL_GET_PAGES)
            data = cursor.fetchall()
            
            for row in data:
                page = Page(*row)
                if page.hashTLSH != "-":
                    pages.append(page)
            self.disconnect()
        except mysql.connector.errors.ProgrammingError as error:
            logger.warning(error)

        return pages

    def get_all_pages_hash(self, algorithm):
        pages = []
        try:
            self.connect()
            cursor = self.conn.cursor(dictionary=True)
            cursor.execute(SQL_GET_PAGES)
            data = cursor.fetchall()

            for row in data:
                page = row.get(algorithm)
                if page != "-":
                    pages.append(page)
            self.disconnect()
        except mysql.connector.errors.ProgrammingError as error:
            logger.warning(error)

        return pages
    
    def get_all_winmodules(self, algorithm):
        winmodules = []
        try:
            self.connect()
            cursor = self.conn.cursor(dictionary=True)
            cursor.execute(SQL_GET_PAGES)
            data = cursor.fetchall()

            for row in data:
                hash = row.get(algorithm)
                if hash != "-":
                    cursor.execute(SQL_GET_MODULE_FROM_HASH, (algorithm, hash))
                    winmodules.append(winmodules)

            self.disconnect()
        except mysql.connector.errors.ProgrammingError as error:
            logger.warning(error)

        return pages


    def get_pages_limit(self, limit):
        pages = []
        try:
            self.connect()
            cursor = self.conn.cursor()
            cursor.execute(SQL_GET_PAGES_LIMIT, (limit,))
            data = cursor.fetchall()
            
            for row in data:
                page = Page(*row)
                if page.hashTLSH != "-":
                    pages.append(page)
            self.disconnect()
        except mysql.connector.errors.ProgrammingError as error:
            logger.warning(error)

        return pages
    
    def get_module_from_hash(self, hash):
        self.connect()
        cursor = self.conn.cursor()
        cursor.execute(SQL_GET_MODULE_FROM_HASH, (hash,))
        row = cursor.fetchone()

        return row[0]
