class Node:
    def __init__(self, id):
        self.id = id
        self.layer = 0
        self.neighbors = [[]]

    def set_max_layer(self, max_layer):
        self.layer = max_layer
        self.neighbors = [[] for _ in range(max_layer+1)]
        
    def add_neighbor(self, layer, neighbor):
        self.neighbors[layer].append(neighbor)

    def calculate_similarity(self, other_node):
        raise NotImplementedError

    def print_neighbors(self):
        string = ""
        for layer in self.neighbors:
            string += "["
            for i in range (0, len(layer)):
                string += str(layer[i].id) + ","
            string += "], "
        return string

    def __str__(self):
        return "Node ID: " + str(self.id) + ", Neighbors: " + self.print_neighbors()