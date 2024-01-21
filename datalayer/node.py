from datalayer.errors import HNSWLayerError

class Node:
    def __init__(self, id):
        self._id = id
        self._layer = 0
        self._neighbors = []

    def set_max_layer(self, max_layer):
        self._layer = max_layer
        self._neighbors = [set() for _ in range(max_layer + 1)]
        
    def add_neighbor(self, layer, neighbor):
        try:
            self._neighbors[layer].add(neighbor)
        except:
            raise HNSWLayerError

    def remove_neighbor(self, layer, neighbor):
        try:
            self._neighbors[layer].remove(neighbor)
        except: # raised if not found
            raise HNSWLayerError

    def get_neighbors_at_layer(self, layer):
        try:
            return self._neighbors[layer]
        except:
            raise HNSWLayerError

    def set_neighbors_at_layer(self, layer, neighbors: set):
        try:
            self._neighbors[layer] = neighbors
        except:
            raise HNSWLayerError

    def calculate_similarity(self, other_node):
        raise NotImplementedError

    def print_neighbors(self):
        string = ""
        for idx, layer in enumerate(self._neighbors):
            string += f"L{idx}: ["
            for _node in layer:
                string += str(_node._id) + ","
            string += "], "
        return string

    def __str__(self):
        return "Node ID: " + str(self._id) + ", Neighbors: " + self.print_neighbors()
    
    def __repr__(self): # for printing while iterating Node data structures
        return "<" + str(self) + ">"

    # getters
    def get_id(self):
        return self._id
    
    def get_layer(self):
        return self._layer
		
    def get_neighbors(self):
        return self._neighbor
