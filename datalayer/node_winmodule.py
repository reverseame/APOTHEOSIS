from node_hash import HashNode

class WinModuleHashNode(HashNode):
    def __init__(self, id, hash_algorithm, module):
        super().__init__(id, hash_algorithm)
        self.module = module
        self.hash_algorithm = hash_algorithm
    
    def __lt__(self, other): # Hack for priority queue
        return False