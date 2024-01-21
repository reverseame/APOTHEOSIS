import ssdeep
from hash_algorithm import HashAlgorithm
from datalayer.score_trend import ScoreTrend

class SSDEEPHashAlgorithm(HashAlgorithm):
    @classmethod 
    def compare(hash1, hash2):
        return (ssdeep.compare(hash1, hash2) - 100) * -1

    @classmethod
    def get_score_trend(self):
        return ScoreTrend.ASCENDING

