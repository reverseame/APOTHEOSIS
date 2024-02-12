import ssdeep

from datalayer.hash_algorithm.hash_algorithm import HashAlgorithm
from datalayer.hash_algorithm.score_trend import ScoreTrend
from datalayer.errors import CharHashValueNotInAlphabetError

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
        total += 3 # symbols are ':', '+', and '/'
        return total
    
    def is_hexhash(cls) -> bool:
        return False

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
        
        _shift += ord('z') - ord('a') + 1
        # symbols are ':', '+' or '/'
        symbols = [':', '+', '/']
        for idx, symb in enumerate(symbols):
            if ch == symb:
                return _shift + idx

        raise CharHashValueNotInAlphabetError(ch)

    @classmethod
    def is_spatial(cls):
        return False # is a similarity metric
