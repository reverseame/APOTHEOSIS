import numpy as np
import pickle
import time
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.getLogger('pickle').setLevel(logging.WARNING)
logging.getLogger('numpy').setLevel(logging.WARNING)
logging.getLogger('time').setLevel(logging.WARNING)

class HNSW:
    def __init__(self, M, ef, Mmax, Mmax0):
        self.found_nearest_elements = []
        self.M = M # Enlaces por nodo
        self.Mmax = Mmax # Enlaces máximos por nodo
        self.Mmax0 = Mmax0 # Enlaces máximos por nodo en la capa 9.
        self.ef = ef
        self.m_L = 1.0 / np.log(self.M)
        self.enter_point = None  # Donde comienza la búsqueda
        self.data = []

    def add_node(self, new_node):
        add_node_init_time = time.time()
        new_node_layer = int(np.floor((-np.log(np.random.uniform(0,1))) * self.m_L))  # Obtener capa a la que pertenece el nuevo nodo

        if self.enter_point == None: # Es el primer nodo que se añade
            new_node.set_max_layer(new_node_layer)
            self.enter_point = new_node
            self.data.append(new_node)
            return

        new_node.set_max_layer(new_node_layer)

        enter_point = self.enter_point

        # Bajar desde el entry point hasta la capa del nuevo nodo...
        for layer in range(self.enter_point.layer, new_node_layer+1, -1):
            currently_found_nn = self.search_layer(new_node, [enter_point], 1, layer)
            if len(currently_found_nn) > 0:
                enter_point = self.find_nearest_element(new_node, currently_found_nn)

        enter_point = [enter_point] 

        #logger.info(f"Adding node {new_node.id} in layer {new_node_layer}")
        # Insertar el nodo desde la capa del nuevo nodo hasta la capa 0.
        for layer in range(min(self.enter_point.layer, new_node_layer), -1, -1):
            currently_found_nn = self.search_layer(new_node, enter_point, self.ef, layer)
            new_neighbors = self.select_neighbours(new_node, currently_found_nn, self.M)
            #logger.debug(f"Nearest neighbors for new node {new_node.id} at layer {layer}: {[n.id for n in new_neighbors]}")
            for neighbor in new_neighbors: # Conectar ambos nodos bidireccionalmente.
                #print(f"Conectando {new_node.id} - {neighbour.id}")
                neighbor.add_neighbor(layer, new_node)
                new_node.add_neighbor(layer, neighbor)
            
            mmax = self.Mmax0 if layer == 0 else self.Mmax

            for neighbor in new_neighbors: # Shrink (cuando hemos superado el límite Mmax)
                if (len(neighbor.neighbors[layer]) > mmax): # CUIDADO! Distinguir aqui con level 0
                    neighbor.neighbors[layer] = self.select_neighbours(neighbor, neighbor.neighbors[layer], self.Mmax)
                    #logger.debug(f"Node {neighbor.id} has exceeeded Mmax. New neigbors reasigned: {[n.id for n in neighbor.neighbors[layer]]}")

            
            enter_point = currently_found_nn
        self.data.append(new_node)

        if new_node.layer > self.enter_point.layer:
            self.enter_point = new_node
        add_node_elapsed_time = time.time() - add_node_init_time
        #logger.info(f'Node {new_node.id} was added in layer {new_node_layer} and took {add_node_elapsed_time}s')

        
    def search_layer(self, node, ep, ef, layer):
        visited_elements = ep.copy()
        candidates = ep.copy()
        currently_found_nearest_neighbours = ep.copy()
        while len(candidates) > 0:
            closest_node = self.find_nearest_element(node, candidates)
            #print(f"Closest node of {node.id} in LAYER: {layer} is {closest_node.id}")
            furthest_node = self.find_furthest_element(node, candidates)
            candidates.remove(closest_node)
            if closest_node.calculate_similarity(node) > furthest_node.calculate_similarity(node):
                break
            
            for neighbour in closest_node.neighbors[layer]:
                if neighbour not in visited_elements:
                    visited_elements.append(neighbour)
                    furthest_node = self.find_furthest_element(node, currently_found_nearest_neighbours)
                    if (neighbour.calculate_similarity(node) < furthest_node.calculate_similarity(node) or len(currently_found_nearest_neighbours) < ef):
                        candidates.append(neighbour)
                        currently_found_nearest_neighbours.append(neighbour)
                        if (len(currently_found_nearest_neighbours) > ef):
                            currently_found_nearest_neighbours.remove(self.find_furthest_element(node, currently_found_nearest_neighbours))

        return currently_found_nearest_neighbours
    
    
    # Obtener los M vecionas más cercanos
    def select_neighbours(self, new_node, candidates, M):
        #print(f"Candidatos: {[n.id for n in candidates]}")
        nearest_neighbours = sorted(candidates, key=lambda obj: obj.calculate_similarity(new_node))
        #print(f"Vecinos cercandos: {[n.id for n in nearest_neighbours[:M]]}")
        return nearest_neighbours[:M]

    def find_nearest_element(self, node, nodes): # Mezclar estas funciones
        nearest_distance = float('inf')
        nearest_element = None
        for n in nodes:
            distance = node.calculate_similarity(n)
            if(distance < nearest_distance):
                nearest_distance = distance
                nearest_element = n
        
        return nearest_element

    def find_furthest_element(self, node, nodes):
        furthest_distance = 0
        furthest_element = None
        for n in nodes:
            dis = node.calculate_similarity(n)
            if(dis >= furthest_distance):
                furthest_distance = dis
                furthest_element = n
        
        return furthest_element

    #def distance(self, node1, node2):
    #    return abs(node2.id - node1.id)

    def get_distances(self, node, nodes):
        distances = []
        for n in nodes:
            distances.append(node.calculate_similarity(n))
        return distances

    def knn_search(self, query, k, ef):
        current_nearest_elements = []
        enter_point = [self.enter_point]
        for layer in range(self.enter_point.layer, 0, -1): # Bajar hasta capa 1
            current_nearest_elements = self.search_layer(query, enter_point, 1, layer)
            enter_point = [self.find_nearest_element(query, current_nearest_elements)]
        current_nearest_elements = self.search_layer(query, enter_point, ef, 0)
        return self.select_neighbours(query, current_nearest_elements, k)
    
    def dump(self, file):
        with open(file, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file):
        with open(file, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected an instance of {cls.__name__}, but got {type(obj).__name__}")
        return obj
    
    def __str__(self):
        string = ""
        for node in self.data:
            string += f"    {node}\n"

        return string + f"Enter point: {self.enter_point}"