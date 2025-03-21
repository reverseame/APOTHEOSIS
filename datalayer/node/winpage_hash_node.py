#TODO docstring
from datalayer.node.hash_node import HashNode
from datalayer.hash_algorithm.hash_algorithm import HashAlgorithm
from datalayer.database.module import Module
from datalayer.db_manager import DBManager
from common.constants import *
from common.errors import NodeUnsupportedAlgorithm

class WinPageHashNode(HashNode):
    def __init__(self, id, hash_algorithm: HashAlgorithm, module: Module=None, db_manager: DBManager=None):
        super().__init__(id, hash_algorithm)
        self._real_module = module
        self._db_manager = db_manager

    @property
    def _module(self):
        if not self._real_module and self._db_manager is not None:
            self._real_module = self._db_manager.get_winpage_module_by_id(self._hash_algorithm, self._id)
        return self._real_module

    def __lt__(self, other): # Hack for priority queue. TODO: not needed here?
        return False

    def get_module(self):
        return self._module
    
    def get_draw_features(self):
        return {"module_names": { self._id: self._module.original_filename + " " + self._module.file_version},
                "module_version": {self._id: self._module.file_version},
                "os_version": {self._id: self._module.os.version}
                }
    
    def as_dict(self):
        node_dict = super().as_dict()
        if self._module:
            node_dict.update({
                "module_id": self._module.id,
                "file_version": self._module.file_version,
                "original_filename": self._module.original_filename,
                "internal_filename": self._module.internal_filename,
                "product_name": self._module.product_name,
                "company_name": self._module.company_name,
                "legal_copyright": self._module.legal_copyright,
                "classification": self._module.classification,
                "size": self._module.size,
                "base_address": self._module.base_address,
            })

            if self._module.os:
                node_dict.update(self._module.os.as_dict())

        return node_dict

    def internal_serialize(self):
        id_bytes = self.get_id().encode('utf-8')
        if len(id_bytes) > HASH_SIZE:
            raise ValueError(f"ID too long - must be <= 144 bytes when encoded")
        return id_bytes.ljust(HASH_SIZE, b'\0') 

    @classmethod
    def internal_load(cls, f):
        bpage_id = f.read(HASH_SIZE)
        return bpage_id, bpage_id.decode('utf-8').rstrip('\x00')

    @classmethod
    def create_node_from_DB(cls, db_manager, hash_id, hash_algorithm, lazy=True):
        new_node = WinPageHashNode(hash_id, hash_algorithm, None, db_manager)
        if not lazy:
            new_node._module # Force load module from database
        return new_node

    @classmethod
    def internal_data_needs_DB(cls) -> bool:
        return True # we have some data necessary to retrieve from the DB
                    # to load a WinPageHashNode from an Apotheosis file

    def is_equal(self, other):
        return other._id == self._id

