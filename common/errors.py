
# User-defined exceptions
class ApotheosisUnsupportedNodeObjectError(Exception):
    pass
class ApotheosisUnsupportedDistanceAlgorithmError(Exception):
    pass
class ApotheosisUnmatchDistanceAlgorithmError(Exception):
    pass
class ApotheosisIsEmptyError(Exception):
    pass
class ApotFileFormatUnsupportedError(Exception):
    pass
class ApotFileReadError(Exception):
    def __init__(self, text, *args):
        super(ApotFileReadError, self).__init__(text, *args)
        self.text = text

# HNSW-related exceptions
class HNSWLayerDoesNotExistError(Exception):
    pass

class HNSWEmptyLayerError(Exception):
    pass

class HNSWIsEmptyError(Exception):
    pass

class HNSWUndefinedError(Exception):
    pass

class HNSWUnmatchDistanceAlgorithmError(Exception):
    pass

# node-related errors
class NodeLayerError(Exception):
    pass

class NodeNotFoundError(Exception):
    pass

class NodeAlreadyExistsError(Exception):
    pass

class NodeUnsupportedAlgorithm(Exception):
    pass

# database-related errors
class HashValueNotInDBError(Exception):
    pass
class PageIdValueNotInDBError(Exception):
    pass

# hash algorithm errors
class CharHashValueNotInAlphabetError(Exception):
    def __init__(self, text, *args):
        super(CharHashValueNotInAlphabetError, self).__init__(text, *args)
        self.text = text
