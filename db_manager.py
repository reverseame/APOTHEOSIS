from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from operating_system import OS
import yaml
import base
import operating_system
import module
import page
from node_winmodules import WinmodulesHashNode
from tlsh_algorithm import TLSHHashAlgorithm
from ssdeep_algorithm import SSDEEPHashAlgorithm

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

    def get_winmodules(self, algorithm):
        winmodules = []
        for os in self.session.query(OS).all():
            for module in os.modules:
                for page in module.pages:
                    if algorithm == TLSHHashAlgorithm and page.hashTLSH != "-":
                        winmodules.append(WinmodulesHashNode(page.hashTLSH, TLSHHashAlgorithm, module))
                    elif algorithm == SSDEEPHashAlgorithm and page.hashSSDEEP != "-":
                        winmodules.append(WinmodulesHashNode(page.hashSSDEEP, SSDEEPHashAlgorithm, module))

        return winmodules


