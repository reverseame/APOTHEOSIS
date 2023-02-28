from node import Node

class HashNode(Node):
    def __init__(self, id, hashAlgorithm):
        super().__init__(id)
        self.hashAlgorithm = hashAlgorithm

    def calculate_similarity(self, other_node):
        return self.hashAlgorithm.compare(self.id, other_node.id)