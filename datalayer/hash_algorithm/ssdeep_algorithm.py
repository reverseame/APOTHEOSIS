import ssdeep
from datalayer.hash_algorithm.hash_algorithm import HashAlgorithm
from datalayer.hash_algorithm.score_trend import ScoreTrend

class SSDEEPHashAlgorithm(HashAlgorithm):
    @classmethod 
    def compare(cls, hash1, hash2):
        return (ssdeep.compare(hash1, hash2) - 100) * -1
    
    @classmethod
    def get_max_hash_alphalen(cls):
        # hash alphabet: digits + lower case + upper case + symbols
        total  = ord('9') - ord('0') + 1
        total += ord('Z') - ord('A') + 1
        total += ord('z') - ord('a') + 1
        total += 2 # symbols are '+' and '/'
        return total

    @classmethod
    def map_to_index(cls, ch) -> int:
        # hash alphabet: digits + lower case + upper case + symbols
        if '0' <= ch and ch <= '9':
             return ord(ch) - ord('0')

        _shift = ord('9') - ord('0') + 1
        if 'A' <= ch and ch <= 'Z':
            return ord(ch) - ord('A') + _shift

        _shift += ord('Z') - ord('A') + 1
        if 'a' <= ch and ch <= 'z':
            return ord(ch) - ord('a') + _shift
        
        # symbols are '+' or '/'
        _shift += ord('z') - ord('a') + 1
        if ch == '+':
            return _shift
        if ch == '/':
            return _shift + 1

        return -1 # will provoke an exception

    @classmethod
    def is_spatial(cls):
        return False # is a similarity metric
