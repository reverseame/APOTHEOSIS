import json
from datalayer.node.node import Node
from datalayer.hash_algorithm.score_trend import ScoreTrend
from datalayer.hash_algorithm.hash_algorithm import HashAlgorithm

class HashNode(Node):
    def __init__(self, id, hash_algorithm: HashAlgorithm):
        super().__init__(id)
        self._hash_algorithm = hash_algorithm
    
    @property
    def score_trend(self):
        return self._hash_algorithm.get_score_trend()

    def calculate_similarity(self, other_node):
        return self._hash_algorithm.compare(self._id, other_node._id)
    
    # checks if n2 is closer than n1 to self
    def n2_closer_than_n1(self, n1=None, n2=None):
        score_n1 = self.calculate_similarity(n1)
        score_n2 = self.calculate_similarity(n2)
        if not self._hash_algorithm.is_spatial(): # similarity metric
            return score_n1 < score_n2, score_n1, score_n2
        else: # distance metric
            return score_n2 < score_n1, score_n1, score_n2
        
    def n1_above_threshold(self, n1=None, threshold=0):
        score = self.calculate_similarity(n1)
        if not self._hash_algorithm.is_spatial(): # similarity metric
            return score < threshold, score
        else: # distance metric
            return threshold < score, score
    
    def __lt__(self, other): # Hack for priority queue
        return False
    
    def to_JSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, 
            sort_keys=True, indent=4)

    def get_distance_algorithm(self):
        return self._hash_algorithm
    
    def as_dict(self):
        return {
            "id": self._id,
            "hash_algorithm": self._hash_algorithm.__name__
        }

