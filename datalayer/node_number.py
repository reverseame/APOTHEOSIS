from node import Node

class NumberNode(Node):
    def __init__(self, id):
        super().__init__(id)

    def calculate_similarity(self, other_node):
        return abs(self.id - other_node.id)