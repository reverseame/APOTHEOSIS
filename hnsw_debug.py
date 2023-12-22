import numpy as np
import random
import pickle
import time
import logging
import heapq
#import page
import os
from score_trend import ScoreTrend

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.getLogger('pickle').setLevel(logging.WARNING)
logging.getLogger('numpy').setLevel(logging.WARNING)
logging.getLogger('time').setLevel(logging.WARNING)

# Score-trend para heapq push

class HNSW:
    def __init__(self, M, ef, Mmax, Mmax0):
        self.found_nearest_elements = []
        self.M = M
        self.Mmax = Mmax # Max links per node
        self.Mmax0 = Mmax0 # ax links per node at layer 0 
        self.ef = ef
        self.m_L = 1.0 / np.log(self.M)
        self.enter_point = None 
        self.queue_multiplier = None

    def add_node(self, new_node):
        """
        Adds a new node to the HNSW index.
        """
        new_node_layer = int(-np.log(random.uniform(0,1)) * self.m_L) // 1  # Get the layer the new node belongs

        if not self.enter_point: # It's the first node being added
            print("This is the enter point!")
            new_node.set_max_layer(new_node_layer)
            self.enter_point = new_node
            if new_node.hash_algorithm.is_score_trend(ScoreTrend.ASCENDING): 
                self.queue_multiplier = 1
            else:
                self.queue_multiplier = -1
            return

        new_node.set_max_layer(new_node_layer)

        enter_point = self.enter_point  # Start searching from the enter point
        # Descend from the entry point to the layer of the new node...
        for layer in range(self.enter_point.layer, new_node_layer+1, -1):
            print(f"Searching at layer {layer}")
            currently_found_nn = self.search_layer_knn(new_node, [enter_point], 1, layer) # Search for the closest node in that layer
            if len(currently_found_nn) > 0:
                #print(f"Me quedo con: {currently_found_nn.get(0)}")
                enter_point = self.find_nearest_element(new_node, currently_found_nn)

        self.insert_node_layers(new_node, [enter_point], new_node_layer)
        print("===========================================================================")


    def delete_node(self, node_to_delete):
        """
        Delete an existing node of the HNSW index.
        """

        enter_point = self.enter_point  # Start searching from the enter point
        # Descend from the entry point to the layer of the new node...
        for layer in range(self.enter_point.layer, 0, -1):
            found_node = self.search_layer_knn(node_to_delete, [enter_point], 1, layer) # Search the closest node in that layer
            if len(found_node) > 0 and found_node.id == node_to_delete.id: # If node is found...
                node_to_delete_neighbors = found_node.neigbors
                for n in node_to_delete_neighbors: # Connect both nodes bidirectionally
                    n.remove_neighbor(found_node) # Delete de link between the neighbor and the node to delete.
                    for n2 in node_to_delete_neighbors: # Create links between node to delete neighbor's
                        if n != n2:
                            n.add_neighbor(layer, n2)
                            n2.add_neighbor(layer, n)
                    mmax = self.Mmax0 if layer == 0 else self.Mmax # Shrink (if Mmax/Mmax0 exceeded)
                    if (len(n.neighbors[layer]) > mmax):
                        n.neighbors[layer] = self.select_neighbours(n, n.neighbors[layer], self.Mmax)

        if node_to_delete == self.enter_point:
            return False
    
    def insert_node_layers(self, new_node, enter_point, new_node_layer):
        """
        Insert the node from the assigned layer of the new node to layer 0.
        """
        print(f"&&&&&&&&&&&&&&&&&&&&& FROM LAYER {new_node_layer} &&&&&&&&&&&&&&&&&&&&&")
        min_layer = min(self.enter_point.layer, new_node_layer)
        for layer in range(min_layer, -1, -1):
            currently_found_nn = self.search_layer_knn(new_node, enter_point, self.ef, layer)
            new_neighbors = self.select_neighbours(new_node, currently_found_nn, self.M)
            for neighbor in new_neighbors: # Connect both nodes bidirectionally
                neighbor.add_neighbor(layer, new_node)
                new_node.add_neighbor(layer, neighbor)
            
            mmax = self.Mmax0 if layer == 0 else self.Mmax

            for neighbor in new_neighbors: # Shrink (when we have exceeded the Mmax limit)
                if (len(neighbor.neighbors[layer]) > mmax):
                    neighbor.neighbors[layer] = self.select_neighbours(neighbor, neighbor.neighbors[layer], self.Mmax)
                    #print(f"Node {neighbor.id} has exceeeded Mmax. New neigbors reasigned: {[n.id for n in neighbor.neighbors[layer]]}")

            enter_point.extend(currently_found_nn)
        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
    
    def search_layer_knn(self, node_query, enter_points, ef, layer):
        """
        Perform k-NN search in a specific layer of the graph.
        """
        print(f"     ---- Performing seach at layer {layer}. Querying {node_query.id[-3:]} with ep {enter_points[0].id[-3:]} ----")
        visited_elements = set(enter_points)
        candidates = []
        currently_found_nearest_neighbors = set(enter_points)

        # Initialize the priority queue with the existing candidates
        for candidate in enter_points:
            distance = candidate.calculate_similarity(node_query)
            heapq.heappush(candidates, (distance*self.queue_multiplier, candidate))

        furthest_node = self.find_furthest_element(node_query, currently_found_nearest_neighbors)
        while len(candidates) > 0:
            # Get the closest node from our candidates list
            _, closest_node = heapq.heappop(candidates)
            print(f"    Current candidates: {candidates}")
            # Check if the closest node from the candidates list is closer than the furthest node from the currently_found_nearest_neighbors list            
            if node_query.who_is_closer(furthest_node, closest_node): 
                print(f"    Break condition!! Furthest:{node_query.calculate_similarity(furthest_node)} < Closest:{node_query.calculate_similarity(closest_node)}")
                break
            else:
                print(f"    Bnreak condition!! Furthest:{node_query.calculate_similarity(furthest_node)} > Closest: {node_query.calculate_similarity(closest_node)}")
            
            #if (len(closest_node.neighbors[layer]) > 0):
            #    self.actual_check(node_query, closest_node.neighbors[layer])
            # Add new candidates to the priority queue
            for neighbor in closest_node.neighbors[layer]:
                if neighbor not in visited_elements:
                    visited_elements.add(neighbor)
                    distance = neighbor.calculate_similarity(node_query)
                    # If the distance is smaller than the furthest node we have in our list, replace it in our list
                    if (node_query.who_is_closer(neighbor, furthest_node)):
                        print(f"    Will add to CFNN node {neighbor.id[-3:]} -> {distance} < {node_query.calculate_similarity(furthest_node)}")
                        if (len(currently_found_nearest_neighbors) < ef):
                            heapq.heappush(candidates, (distance*self.queue_multiplier, neighbor))
                            currently_found_nearest_neighbors.add(neighbor)
                            if len(currently_found_nearest_neighbors) > ef:
                                currently_found_nearest_neighbors.remove(self.find_furthest_element(node_query, currently_found_nearest_neighbors))
        self.print_cfnn(node_query, currently_found_nearest_neighbors)
        print("      -------------------------------------------------------------------")
        return currently_found_nearest_neighbors

    def actual_check(self, query, neighbors):
        print("     /\\ Actual check /\\")
        pairs = []
        
        for n in neighbors:
            distance = n.calculate_similarity(query)
            pairs.append((n.id[:-3], distance))

        [print(f"   {n}") for n in pairs]


        min_tuple = min(pairs, key=lambda x: x[1])
        print(f"    Closest neighbor is: {min_tuple}. Continue with this one!")
        print("     /\\ END Actual check /\\")

    def print_cfnn(self, node_query, currently_found_nearest_neighbors):
        print("     /\\ CFNN check /\\")
        pairs = []

        for n in currently_found_nearest_neighbors:
            distance = n.calculate_similarity(node_query)
            pairs.append((n.id[:-3], distance))

        [print(f"   {n}") for n in pairs]
        min_tuple = min(pairs, key=lambda x: x[1])
        print(f"    Closest CFNN is: {min_tuple}. Continue with this one!")

        print("     /\\ END CFNN check /\\")


    def search_layer_percentage(self, node_query, enter_points, percentage):
        visited_elements = set(enter_points)
        candidates = []
        currently_found_nearest_neighbors = set(enter_points)
        final_elements = set()

        # Initialize the priority queue with the existing candidates
        for candidate in enter_points:
            distance = candidate.calculate_similarity(node_query)
            heapq.heappush(candidates, (distance*self.queue_multiplier, candidate))

        furthest_node = self.find_furthest_element(node_query, currently_found_nearest_neighbors)
        while len(candidates) > 0:
            # Get the closest node from our candidates list
            _, closest_node = heapq.heappop(candidates)

            # Check if the closest node from the candidates list is closer than the furthest node from the list            
            if node_query.who_is_closer(closest_node, furthest_node): 
                break # All elements from currently_found_nearest_neighbors have been evaluated

            # Add new candidates to the priority queue
            for neighbor in closest_node.neighbors[0]:
                if neighbor not in visited_elements:
                    visited_elements.add(neighbor)
                    distance = neighbor.calculate_similarity(node_query)
                    # If the neighbor's distance satisfies the threshold, it joins the list.
                    if (node_query.who_is_closer(neighbor, furthest_node)):
                        heapq.heappush(candidates, (distance*self.queue_multiplier, neighbor))
                        if (distance > percentage):
                            final_elements.add(neighbor)

        return final_elements

    def select_neighbours(self, new_node, candidates, M):
        """Get the M nearest neighbors.
        """
        nearest_neighbours = sorted(candidates, key=lambda obj: obj.calculate_similarity(new_node))
        return nearest_neighbours[:M]
    
    def find_furthest_element(self, node, nodes): 
        return min((n for n in nodes if n != node), key=lambda n: node.calculate_similarity(n), default=None)

    def find_nearest_element(self, node, nodes):
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
            pickle.dump(self, f, protocol=pickle.DEFAULT_PROTOCOL)

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

    def knn_search(self, query, k, ef=0): 
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

    def percentage_search(self, query, percentage):
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
        for layer in range(self.enter_point.layer, 0, -1): # Descend to layer 1
            current_nearest_elements = self.search_layer_knn(query, enter_point, 1, layer)
            enter_point = [self.find_nearest_element(query, current_nearest_elements)]
        
        return self.search_layer_percentage(query, enter_point, percentage)
    

