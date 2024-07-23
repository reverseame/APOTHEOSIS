from common.errors import NodeLayerError

# HNSW Node class definition
#TODO add docstring
class Node:
    def __init__(self, id):
        self._id = id
        self._max_layer = 0
        self._neighbors = []

    def set_max_layer(self, max_layer: int):
        self._max_layer = max_layer
        self._neighbors = [set() for _ in range(max_layer + 1)]
        
    def add_neighbor(self, layer: int, neighbor):
        try:
            self._neighbors[layer].add(neighbor)
        except: # raised if not found
            raise NodeLayerError

    def remove_neighbor(self, layer: int, neighbor):
        try:
            self._neighbors[layer].remove(neighbor)
        except: # raised if not found
            raise NodeLayerError

    def get_neighbors_at_layer(self, layer: int):
        try:
            return self._neighbors[layer]
        except:
            raise NodeLayerError

    def set_neighbors_at_layer(self, layer: int, neighbors: set):
        try:
            self._neighbors[layer] = neighbors
        except:
            raise NodeLayerError

    # only in HashNode
    def calculate_similarity(self, other_node):
        raise NotImplementedError
    # only in WinModuleHashNode
    def get_pageids(self):
        raise NotImplementedError

    # to be implemented in final classes
    def internal_serialize(self):
        raise NotImplementedError
    # to be implemented in final classes
    def internal_load(cls, f):
        raise NotImplementedError

    def print_neighbors(self):
        string = ""
        for idx, layer in enumerate(self._neighbors):
            string += f"L{idx}: ["
            for node in layer:
                string += str(node._id) + ","
            string += "], "
        return string

    def __str__(self):
        return "Node ID: " + str(self._id) + ", Neighbors: " + self.print_neighbors()
    
    def as_dict(self):
        return {
            "id": str(self._id),
            "neighbors": self.print_neighbors()
        }
    
    def __repr__(self): # for printing while iterating Node data structures
        return "<" + str(self) + ">"

    # getters
    def get_id(self):
        return self._id
    
    def get_max_layer(self):
        return self._max_layer
		
    def get_neighbors(self):
        return self._neighbors
    
    def get_draw_features(self):
        raise NotImplementedError
