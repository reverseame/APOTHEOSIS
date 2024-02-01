import ssdeep
from datalayer.hash_algorithm.hash_algorithm import HashAlgorithm
from datalayer.hash_algorithm.score_trend import ScoreTrend

class SSDEEPHashAlgorithm(HashAlgorithm):
    @classmethod 
    def compare(cls, hash1, hash2):
        return (ssdeep.compare(hash1, hash2) - 100) * -1

    @classmethod
    def is_spatial(self):
        return False # is a similarity metric
