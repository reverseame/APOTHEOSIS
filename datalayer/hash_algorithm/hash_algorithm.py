from abc import ABC, abstractmethod

class HashAlgorithm(ABC):
    @abstractmethod
    def compare(self, hash1, hash2):
        pass
	
    @abstractmethod
    def get_score_trend(self):
        pass
    
    @abstractmethod
    def get_max_hash_alphalen(self) -> int:
        pass
    
    @abstractmethod
    def map_to_index(self, ch) -> int:
        pass

    @abstractmethod
    def is_spatial(self):
        pass
    # spatial algorithms: distance metric
    # non-spatial: similarity score
