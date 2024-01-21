import json
from datalayer.node import Node
from datalayer.score_trend import ScoreTrend

class HashNode(Node):
    def __init__(self, id, hash_algorithm):
        super().__init__(id)
        self._hash_algorithm = hash_algorithm
    
    @property
    def score_trend(self):
        return self._hash_algorithm.get_score_trend()

    def ascending_trend(self):
        return self.score_trend == ScoreTrend.ASCENDING;

    def calculate_similarity(self, other_node):
        return self._hash_algorithm.compare(self._id, other_node._id)
    
    # checks if n2 is closer than n1 to self
    def n2_closer_than_n1(self, n1=None, n2=None):
        score_n1 = self.calculate_similarity(n1)
        score_n2 = self.calculate_similarity(n2)
        if self.ascending_trend():
            return score_n1 < score_n2, score_n1, score_n2
        else:
            return score_n2 < score_n1, score_n1, score_n2
    
    def __lt__(self, other): # Hack for priority queue
        return False
    
    def to_JSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, 
            sort_keys=True, indent=4)

    def get_hash(self):
        return self._hash_algorithm
