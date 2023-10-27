from node_hash import HashNode

class WinmodulesHashNode(HashNode):
    def __init__(self, id, hash_algorithm, module):
        super().__init__(id, hash_algorithm)
        self.module = module
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