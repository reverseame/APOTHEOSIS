import numpy as np
import random
import pickle
import time
import logging
import heapq

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.getLogger('pickle').setLevel(logging.WARNING)
logging.getLogger('numpy').setLevel(logging.WARNING)
logging.getLogger('time').setLevel(logging.WARNING)

class HNSW:
    def __init__(self, M, ef, Mmax, Mmax0):
        self.found_nearest_elements = []
        self.M = M
        self.Mmax = Mmax # Enlaces máximos por nodo
        self.Mmax0 = Mmax0 # Enlaces máximos por nodo en la capa 0.
        self.ef = ef
        self.m_L = 1.0 / np.log(self.M)
        self.enter_point = None  # Nodo donde comienza la búsqueda

    def add_node(self, new_node):
        """
        Adds a new node to the HNSW index.
        """
        new_node_layer = int(-np.log(random.uniform(0,1)) * self.m_L) // 1  # Get the layer the new node belongs

        if not self.enter_point: # It's the first node being added
            new_node.set_max_layer(new_node_layer)
            self.enter_point = new_node
            return

        new_node.set_max_layer(new_node_layer)

        enter_point = self.enter_point  # Start searching from the enter point
        # Descend from the entry point to the layer of the new node...
        for layer in range(self.enter_point.layer, new_node_layer+1, -1):
            currently_found_nn = self.search_layer_knn(new_node, [enter_point], 1, layer) # Buscamos el nodo más cercano en esa capa
            if len(currently_found_nn) > 0:
                enter_point = self.find_nearest_element(new_node, currently_found_nn)

        self.insert_node_layers(new_node, [enter_point], new_node_layer)

        if new_node.layer > self.enter_point.layer:
            self.enter_point = new_node
    
    def insert_node_layers(self, new_node, enter_point, new_node_layer):
        """
        Insert the node from the assigned layer of the new node to layer 0.
        """
        #logger.info(f"Adding node {new_node.id} from layer {new_node_layer} to layer 0")
        min_layer = min(self.enter_point.layer, new_node_layer)
        for layer in range(min_layer, -1, -1):
            currently_found_nn = self.search_layer_knn(new_node, enter_point, self.ef, layer)
            new_neighbors = self.select_neighbours(new_node, currently_found_nn, self.M)
            #logger.debug(f"Nearest neighbors for new node {new_node.id} at layer {layer}: {[n.id for n in new_neighbors]}")
            for neighbor in new_neighbors: # Connect both nodes bidirectionally
                neighbor.add_neighbor(layer, new_node)
                new_node.add_neighbor(layer, neighbor)
            
            mmax = self.Mmax0 if layer == 0 else self.Mmax

            for neighbor in new_neighbors: # Shrink (when we have exceeded the Mmax limit)
                if (len(neighbor.neighbors[layer]) > mmax):
                    neighbor.neighbors[layer] = self.select_neighbours(neighbor, neighbor.neighbors[layer], self.Mmax)
                    #logger.debug(f"Node {neighbor.id} has exceeeded Mmax. New neigbors reasigned: {[n.id for n in neighbor.neighbors[layer]]}")

            enter_point.extend(currently_found_nn)
    
    def search_layer_knn(self, node_query, enter_points, ef, layer):
        """
        Perform k-NN search in a specific layer of the graph.
        """
        visited_elements = set(enter_points)
        candidates = []
        currently_found_nearest_neighbors = set(enter_points)

        # Initialize the priority queue with the existing candidates
        for candidate in enter_points:
            distance = candidate.calculate_similarity(node_query)
            heapq.heappush(candidates, (distance, candidate))

        furthest_node = self.find_furthest_element(node_query, currently_found_nearest_neighbors)
        while len(candidates) > 0:
            # Get the closest node from our candidates list
            closest_distance, closest_node = heapq.heappop(candidates)

            # Check if the closest node from the candidates list is closer than the furthest node from the list            
            if closest_distance > furthest_node.calculate_similarity(node_query):
                break # All elements from currently_found_nearest_neighbors have been evaluated

            # Add new candidates to the priority queue
            for neighbor in closest_node.neighbors[layer]:
                if neighbor not in visited_elements:
                    visited_elements.add(neighbor)
                    distance = neighbor.calculate_similarity(node_query)
                    # If the distance is smaller than the furthest node we have in our list, replace it in our list
                    if (distance < furthest_node.calculate_similarity(node_query) or len(currently_found_nearest_neighbors) < ef):
                        heapq.heappush(candidates, (distance, neighbor))
                        currently_found_nearest_neighbors.add(neighbor)
                        if len(currently_found_nearest_neighbors) > ef:
                            currently_found_nearest_neighbors.remove(self.find_furthest_element(node_query, currently_found_nearest_neighbors))

        return currently_found_nearest_neighbors

    def search_layer_percentage(self, node_query, enter_points, percentage):
        visited_elements = set(enter_points)
        candidates = []
        currently_found_nearest_neighbors = set(enter_points)
        final_elements = set()

        # Initialize the priority queue with the existing candidates
        for candidate in enter_points:
            distance = candidate.calculate_similarity(node_query)
            heapq.heappush(candidates, (distance, candidate))

        furthest_node = self.find_furthest_element(node_query, currently_found_nearest_neighbors)
        while len(candidates) > 0:
            # Get the closest node from our candidates list
            closest_distance, closest_node = heapq.heappop(candidates)

            # Check if the closest node from the candidates list is closer than the furthest node from the list            
            if closest_distance > furthest_node.calculate_similarity(node_query):
                break # All elements from currently_found_nearest_neighbors have been evaluated

            # Add new candidates to the priority queue
            for neighbor in closest_node.neighbors[0]:
                if neighbor not in visited_elements:
                    visited_elements.add(neighbor)
                    distance = neighbor.calculate_similarity(node_query)
                    # If the distance satisfies the threshold, it enters the list.
                    if (distance < furthest_node.calculate_similarity(node_query)):
                        heapq.heappush(candidates, (distance, neighbor))
                        if (distance < percentage):
                            final_elements.add(neighbor)

        return final_elements

    def select_neighbours(self, new_node, candidates, M):
        """Get the M nearest neighbors.
        """
        nearest_neighbours = sorted(candidates, key=lambda obj: obj.calculate_similarity(new_node))
        return nearest_neighbours[:M]
    
    def find_nearest_element(self, node, nodes): # Mezclar estas funciones
        return min((n for n in nodes if n != node), key=lambda n: node.calculate_similarity(n), default=None)

    def find_furthest_element(self, node, nodes):
        return max((n for n in nodes if n != node), key=lambda n: node.calculate_similarity(n), default=None)

    def get_distances(self, node, nodes):
        distances = []
        for n in nodes:
            distances.append(node.calculate_similarity(n))
        return distances
    
    def dump(self, file):
        """
        Saves HNSW structure to disk
        """

        with open(file, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file):
        """
        Restores HNSW structure from disk
        """
        with open(file, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected an instance of {cls.__name__}, but got {type(obj).__name__}")
        return obj

    def knn_search(self, query, k, ef): 
        """
        Performs k-nearest neighbors search using the HNSW algorithm.
        
        Args:
            query: The query node for which to find the nearest neighbors.
            k: The number of nearest neighbors to retrieve.
            ef: The exploration factor controlling the search efficiency.
        
        Returns:
            A list of k nearest neighbor nodes to the query node.
        """
        current_nearest_elements = []
        enter_point = [self.enter_point]
        for layer in range(self.enter_point.layer, 0, -1): # Descend to layer 1
            current_nearest_elements = self.search_layer_knn(query, enter_point, 1, layer)
            enter_point = [self.find_nearest_element(query, current_nearest_elements)]
        current_nearest_elements = self.search_layer_knn(query, enter_point, ef, 0)
        return self.select_neighbours(query, current_nearest_elements, k)

    def percentage_search(self, query, percentage): # AÑADIR OPERATION (LOWER THAN; GREATER THAN). Modificar GT/LS.
        """
            Performs a percentage search to retrieve nodes that satisfy a certain similarity threshold using the HNSW algorithm.
        
        Args:
            query: The query node for which to find the nearest neighbors.
            percentage: The threshold percentage for similarity. Nodes with similarity greater than or less than to this
                    threshold will be returned.
        
        Returns:
            A list of nearest neighbor nodes that satisfy the specified similarity threshold.

        """
        current_nearest_elements = []
        enter_point = [self.enter_point]
        for layer in range(self.enter_point.layer, 0, -1): # Bajar hasta capa 1
            current_nearest_elements = self.search_layer_knn(query, enter_point, 1, layer)
            enter_point = [self.find_nearest_element(query, current_nearest_elements)]
        
        return self.search_layer_percentage(query, enter_point, percentage)
    

