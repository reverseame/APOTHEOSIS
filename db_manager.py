from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datalayer.database.operating_system import OS
import yaml
import datalayer.database.base
import datalayer.database.operating_system
import datalayer.database.module
import datalayer.database.page
from datalayer.node.winmodule_hash_node import WinModuleHashNode
from datalayer.hash_algorithm.tlsh_algorithm import TLSHHashAlgorithm
from datalayer.hash_algorithm.ssdeep_algorithm import SSDEEPHashAlgorithm

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

    def get_winmodules(self, algorithm, _limit: int = None):
        winmodules = []

        _query = self.session.query(OS).all()

        for os in _query:
            for module in os.modules:
                for page in module.pages:
                    if _limit is not None and len(winmodules) >= _limit:
                        return winmodules
                    if algorithm == TLSHHashAlgorithm and page.hashTLSH != "-":
                        winmodules.append(WinModuleHashNode(page.hashTLSH, TLSHHashAlgorithm, module))
                    elif algorithm == SSDEEPHashAlgorithm and page.hashSSDEEP != "-":
                        winmodules.append(WinModuleHashNode(page.hashSSDEEP, SSDEEPHashAlgorithm, module))

        return winmodules

    def close(self):
        self.session.close()

