import tlsh
from datalayer.hash_algorithm.hash_algorithm import HashAlgorithm
from datalayer.hash_algorithm.score_trend import ScoreTrend

class TLSHHashAlgorithm(HashAlgorithm):
    @classmethod
    def compare(cls, hash1, hash2):
        return tlsh.diff(hash1, hash2)

    @classmethod
    def get_max_hash_alphalen(self):
        total = ord('9') - ord('0') + 1
        total += ord('Z') - ord('A') + 1
        total += ord('z') - ord('a') + 1
        return total

    @classmethod
    def is_spatial(self):
        return True # is a distance metric
