from abc import ABC, abstractmethod

class HashAlgorithm(ABC):
    @abstractmethod
    def compare(self, hash1, hash2):
        pass
