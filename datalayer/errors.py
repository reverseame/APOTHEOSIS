
# User-defined exceptions
class HNSWLayerError(Exception):
    pass

class HNSWUndefinedError(Exception):
    pass

class HNSWUnmatchDistanceAlgorithmError(Exception):
    pass

class NodeNotFoundError(Exception):
    pass

class NodeAlreadyExistsError(Exception):
    pass
