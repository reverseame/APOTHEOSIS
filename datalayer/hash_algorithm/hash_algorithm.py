from abc import ABC, abstractmethod

class HashAlgorithm(ABC):
    # necessary for storing a HashAlgorithm in a HNSW of HashNode 
    @classmethod
    @abstractmethod
    def compare(cls, hash1, hash2):
        pass
	
    @classmethod
    @abstractmethod
    def get_score_trend(cls):
        pass
    
    @classmethod
    @abstractmethod
    def is_spatial(cls):
        pass
    # spatial algorithms: distance metric
    # non-spatial: similarity score
    
    # necessary for storing a HashAlgorithm in a TrieHash 
    @classmethod
    @abstractmethod
    def get_max_hash_alphalen(cls) -> int:
        pass
    
    @classmethod
    @abstractmethod
    def map_to_index(cls, ch) -> int:
        pass

