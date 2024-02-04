from abc import ABC, abstractmethod

class HashAlgorithm(ABC):
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
    def get_max_hash_alphalen(cls) -> int:
        pass
    
    @classmethod
    @abstractmethod
    def map_to_index(cls, ch) -> int:
        pass

    @classmethod
    @abstractmethod
    def map_to_charhash(cls, index: int):
        pass
    
    @classmethod
    @abstractmethod
    def is_spatial(cls):
        pass
    # spatial algorithms: distance metric
    # non-spatial: similarity score
