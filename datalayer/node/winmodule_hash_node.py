#TODO docstring
from datalayer.node.hash_node import HashNode
from datalayer.hash_algorithm.hash_algorithm import HashAlgorithm
from datalayer.database.module import Module
from datalayer.database.page import Page

class WinModuleHashNode(HashNode):
    def __init__(self, id, hash_algorithm: HashAlgorithm, module: Module, page: Page):
        super().__init__(id, hash_algorithm)
        self._module = module
        self._page = page

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

