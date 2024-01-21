import tlsh
from datalayer.hash_algorithm import HashAlgorithm
from datalayer.score_trend import ScoreTrend

class TLSHHashAlgorithm(HashAlgorithm):
    @classmethod
    def compare(self, hash1, hash2):
        return tlsh.diff(hash1, hash2)

    @classmethod
    def get_score_trend(self):
        return ScoreTrend.DESCENDING
