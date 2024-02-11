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
