import json
from node import Node

class HashNode(Node):
    def __init__(self, id, hash_algorithm):
        super().__init__(id)
        self.hash_algorithm = hash_algorithm

    def calculate_similarity(self, other_node):
        return self.hash_algorithm.compare(self.id, other_node.id)
    
    def who_is_closer(self, node1, node2):
        score1 = self.calculate_similarity(node1)
        score2 = self.calculate_similarity(node2)
        if self.hash_algorithm.is_greater_trend():
            return score1 > score2
        else:
            return score1 < score2
    
    def __lt__(self, other): # Hack for priority queue
        return False
    
    def to_JSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, 
            sort_keys=True, indent=4)