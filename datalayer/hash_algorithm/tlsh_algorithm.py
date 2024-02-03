import tlsh
from datalayer.hash_algorithm.hash_algorithm import HashAlgorithm
from datalayer.hash_algorithm.score_trend import ScoreTrend

class TLSHHashAlgorithm(HashAlgorithm):
    @classmethod
    def compare(cls, hash1, hash2):
        return tlsh.diff(hash1, hash2)

    @classmethod
    def get_max_hash_alphalen(cls) -> int:
        # hash alphabet: digits + upper case + lower case
        total  = ord('9') - ord('0') + 1
        total += ord('Z') - ord('A') + 1
        total += ord('z') - ord('a') + 1
        return total

    @classmethod
    def map_to_index(cls, ch) -> int:
        # hash alphabet: digits + upper case + lower case
        if '0' <= ch and ch <= '9':
             return ord(ch) - ord('0')

        _shift = ord('9') - ord('0') + 1
        if 'A' <= ch and ch <= 'Z':
            return ord(ch) - ord('A') + _shift

        _shift += ord('Z') - ord('A') + 1
        return ord(ch) - ord('a') + _shift
    
    @classmethod
    def is_spatial(cls):
        return True # is a distance metric
