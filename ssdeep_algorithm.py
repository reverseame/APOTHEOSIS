import ssdeep
from hash_algorithm import HashAlgorithm

class SSDEEPHashAlgorithm(HashAlgorithm):
    def compare(hash1, hash2):
        #print(f"Comparing {hash1} con {hash2}: ")
        #print(ssdeep.compare(hash1, hash2))
        return (ssdeep.compare(hash1, hash2) - 100) * -1

    def is_greater_trend():
        return True
