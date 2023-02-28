import tlsh
from hash_algorithm import HashAlgorithm

class TLSHHashAlgorithm(HashAlgorithm):
    def compare(hash1, hash2):
        return tlsh.diff(hash1, hash2)