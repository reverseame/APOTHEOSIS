from abc import ABC, abstractmethod

class HashAlgorithm(ABC):
    @abstractmethod
    def compare(self, hash1, hash2):
        pass
	
    @abstractmethod
    def get_score_trend(self):
        pass

    """
    # The closer to 100, the more similar
    def ascending_trend_score(self):
        return self._score_trend == ScoreTrend.ASCENDING
    
    # The closer to 0, the more similar 
    def descending_trend_score(self):
        return self._score_trend == ScoreTrend.ASCENDING
	"""
