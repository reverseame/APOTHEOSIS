import tlsh

from datalayer.hash_algorithm.hash_algorithm import HashAlgorithm
from datalayer.hash_algorithm.score_trend import ScoreTrend

from common.errors import CharHashValueNotInAlphabetError

class TLSHHashAlgorithm(HashAlgorithm):
    
    @classmethod
    def compare(cls, hash1, hash2):
        return tlsh.diff(hash1, hash2)

    @classmethod
    def get_max_hash_alphalen(cls) -> int:
        # hash alphabet: hexadecimal bytes + 'T'
        total  = ord('9') - ord('0') + 1
        total += ord('F') - ord('A') + 1 # we consider only upper chars
        total += 1 # value 'T'
        # first values of a TLSH hash may be 'T1' to determine the version
        # old TLSH hashes will have only 70 hex characters
        return total
    
    @classmethod
    def is_hexhash(cls) -> bool:
        return True

    @classmethod
    def map_to_index(cls, ch) -> int:
        # hash alphabet: hexadecimal bytes + 'T'
        if '0' <= ch and ch <= '9':
             return ord(ch) - ord('0')

        _shift = ord('9') - ord('0') + 1
        ch = ch.upper() # upper case only
        if 'A' <= ch and ch <= 'F':
            return ord(ch) - ord('A') + _shift

        _shift += ord('F') - ord('A') + 1
        if ch == 'T':
            return _shift
        
        raise CharHashValueNotInAlphabetError(ch)
    
    @classmethod
    def is_spatial(cls):
        return True # is a distance metric
