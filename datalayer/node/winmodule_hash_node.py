#TODO docstring
from datalayer.node.hash_node import HashNode
from datalayer.hash_algorithm.hash_algorithm import HashAlgorithm
from datalayer.hash_algorithm.tlsh_algorithm import TLSHHashAlgorithm
from datalayer.hash_algorithm.ssdeep_algorithm import SSDEEPHashAlgorithm
from datalayer.database.module import Module
from common.constants import *
from common.errors import NodeUnsupportedAlgorithm

class WinModuleHashNode(HashNode):
    def __init__(self, id, hash_algorithm: HashAlgorithm, module: Module=None):
        super().__init__(id, hash_algorithm)
        self._module = module

    def __lt__(self, other): # Hack for priority queue. TODO: not needed here?
        return False

    def get_module(self):
        return self._module
    
    def get_page(self):
        return self._page

    def get_internal_page_id(self):
        return self._page.id if self._page else 0
    
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
        bpage_id = f.read(I_SIZE)
        return bpage_id, int.from_bytes(bpage_id, byteorder=BYTE_ORDER)

    @classmethod
    def create_node_from_DB(cls, db_manager, hash_id, hash_algorithm):
        new_node = db_manager.get_winmodule_data_by_hash(hash_value=hash_id, algorithm=hash_algorithm)
        if hash_algorithm == TLSHHashAlgorithm:
            new_node._id = new_node._page.hashTLSH
        elif hash_algorithm == SSDEEPHashAlgorithm:
            new_node._id = new_node._page.hashSSDEEP
        else:
            raise NodeUnsupportedAlgorithm # algorithm not supported

        return new_node

    @classmethod
    def internal_data_needs_DB(cls) -> bool:
        return True # we have some data necessary to retrieve from the DB
                    # to load a WinModuleHashNode from an Apotheosis file

    def is_equal(self, other):
        if type(self) != type(other):
            return False
        try:
            same_module = self._module == other._module
            same_page = self._page == other._page
            if not same_module or not same_page:
                return False
            if type(self._hash_algorithm) != type(other._hash_algorithm):
                return False
            # check now the id and the hash, both modules and pages are the same
            equal = self._id == other._id and self._max_layer == other._max_layer and\
                        len(self._neighbors) == len(other._neighbors)
            if not equal:
                return False
            # now, check the neighbors
            for idx, neighs in enumerate(self._neighbors):
                other_pageid = set([node._page.id for node in other._neighbors[idx]])
                self_pageid = set([node._page.id for node in self._neighbors[idx]])
                if other_pageid != self_pageid:
                    return False
            
            return True
        except:
            return False

