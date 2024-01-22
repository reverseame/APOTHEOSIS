import numpy as np
import random
import pickle
import time
import logging
import heapq
import os

# custom exceptions
from datalayer.errors import NodeNotFound
from datalayer.errors import NodeAlreadyExists
from datalayer.errors import HNSWUndefinedError

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.getLogger('pickle').setLevel(logging.WARNING)
logging.getLogger('numpy').setLevel(logging.WARNING)
logging.getLogger('time').setLevel(logging.WARNING)

class HNSW:
    # initial layer will have index 0
    
    def __init__(self, M, ef, Mmax, Mmax0,
                    distance_algorithm=None,
                    heuristic=False, extend_candidates=True, keep_pruned_conns=True):
        self._found_nearest_elements = []
        self._M = M
        self._Mmax = Mmax # max links per node
        self._Mmax0 = Mmax0 # max links per node at layer 0 
        self._ef = ef
        self._mL = 1.0 / np.log(self._M)
        self._enter_point = None
        self._nodes = dict()
        self._heuristic = heuristic
        self._extend_candidates = extend_candidates
        self._keep_pruned_conns = keep_pruned_conns
        self._distance_algorithm = distance_algorithm

    def _insert_node(self, node):
        """
        TODO
        """
        if self._nodes.get(node.get_layer()) is None:
            self._nodes[node.get_layer()] = list()
        
        self._nodes[node.get_layer()].append(node)

    def get_enter_point(self):
        """
        Getter for _enter_point
        """
        return self._enter_point

    def _descend_to_layer(self, query_node, layer=0):
        """
        Given a query_node and a given layer, it goes down to that layer and returns the enter point for query_node
        """
        enter_point = self._enter_point
        for layer in range(self._enter_point.get_layer(), layer, -1): # Descend to the given layer
            logging.debug(f"Descending into layer {layer}, ep: {enter_point}")
            current_nearest_elements = self.search_layer_knn(query_node, [enter_point], 1, layer)
            if len(current_nearest_elements) > 0:
                if enter_point.get_id() != query_node.get_id():
                    # get the nearest element to query node
                    # when removing a node, they can be the sam element. We use the node id to check this fact
                    enter_point = self.find_nearest_element(query_node, current_nearest_elements)
            else: #XXX is this path even feasible?
                logging.warning("No closest neighbor found at layer {}".format(layer))


        return enter_point

    def add_node(self, new_node):
        """
        Adds a new node to the HNSW index. 
        Raises "NodeAlreadyExists" if it already exists a node with the same ID as the one of new_node
        """
        # check if the HNSW distance algorithm is the same as the one associated to the node
        # raise exception otherwise
        #TODO
        
        enter_point = self._enter_point
        # Calculate the layer to which the new node belongs
        new_node_layer = int(-np.log(random.uniform(0,1)) * self._mL) // 1 # l in MY-TPAMI-20
        new_node.set_max_layer(new_node_layer)
        logging.info(f"Inserting new node: {new_node} (assigned level: {new_node_layer})")
        
        if enter_point is not None:
            Lep = enter_point.get_layer()
            
            # Descend from the entry point to the layer of the new node...
            enter_point = self._descend_to_layer(new_node, layer=new_node_layer)

            if enter_point.get_id() == new_node.get_id():
                raise NodeAlreadyExists

            # Insert the new node
            self.insert_node_layers(new_node, [enter_point])

            # Update higher layer, if necessary
            if new_node_layer > Lep:
                self._enter_point = new_node
                logging.info("Setting new node as enter point ... ")

        else:
            self._enter_point = new_node
            logging.info("Setting new node as enter point ... ")
        # store it now
        self._insert_node(new_node)

    def _delete_neighbors_connections(self, node):
        """
        TODO
        """

        # iterate for each layer and for each neighbor in that layer
        # adjust connection of this neighbor to closest neighbors
        pass

    def _adjust_neighbor_connections(self, node):
        """
        TODO
        """
        pass

    def delete_node(self, node_to_delete):
        """
        Delete an existing node of the HNSW index.
        It raises NodeNotFound if node_to_delete is not found in the structure and
        HNSWUndefinedError when no neighbor is found at layer 0
        """
        # check if it is empty
        # TODO raise HNSWIsEmpty
        # check if the HNSW distance algorithm is the same as the one associated to the node
        # raise exception otherwise
        #TODO

        # from the enter_point, reach the node, if exists
        enter_point = self._descend_to_layer(node_to_delete)
        # now checks for the node, if it is in this layer
        found_node = self.search_layer_knn(node_to_delete, [enter_point], 1, 0)
        if len(found_node) == 1:
            found_node = found_node.pop()
            if found_node.get_id() == node_to_delete.get_id():
                logging.debug("Node {node_to_delete} found! Deleting it ...")
                # now delete neighbor connections
                self._delete_neighbors_connections(found_node)
               
                #TODO
                # check if this is the enter point to the structure

                # if so, update enter point to the first closest neighbor (if any)
            else:
                raise NodeNotFound
        else:
            # It should always get one closest neighbor, unless it is empty
            raise HNSWUndefinedError

    def _already_exists(self, query_node, node_list):
        """
        TODO
        """
        for node in node_list:
            if node.get_id() == query_node.get_id():
                return True
        return False
    
    def _shrink_nodes(self, nodes, layer):
        """
        TODO
        """

        mmax = self._Mmax0 if layer == 0 else self._Mmax

        for _node in nodes:
            _list = _node.get_neighbors_at_layer(layer)
            if (len(_list) > mmax):
                _node.set_neighbors_at_layer(layer, self.select_neighbors(node, _list, mmax, layer))
                logging.debug(f"Node {_node.id} exceeded Mmax. New neighbors: {[n.id for n in node.get_neighbors_at_layer(layer)]}")

    def insert_node_layers(self, new_node, enter_point):
        """
        Insert the node from the assigned layer of the new node to layer 0.
        Raises NodeAlreadyExists if the new_node already exists in the HNSW structure
        """
        
        min_layer = min(self._enter_point.get_layer(), new_node.get_layer())
        for layer in range(min_layer, -1, -1):
            currently_found_nn = self.search_layer_knn(new_node, enter_point, self._ef, layer)
            new_neighbors = self.select_neighbors(new_node, currently_found_nn, self._M, layer)
            logging.debug(f"Found nn at L{layer}: {currently_found_nn}")

            if self._already_exists(new_node, currently_found_nn):
                raise NodeAlreadyExists

            # connect both nodes bidirectionally
            for neighbor in new_neighbors: 
                neighbor.add_neighbor(layer, new_node)
                new_node.add_neighbor(layer, neighbor)
                logging.info(f"Connections added at L{layer} between {new_node} and {neighbor}")
            
            # shrink (when we have exceeded the Mmax limit)
            self._shrink_nodes(new_neighbors, layer)
            enter_point.extend(currently_found_nn)
        
    def search_layer_knn(self, query_node, enter_points, ef, layer):
        """
        Perform k-NN search in a specific layer of the graph.
        """
        visited_elements = set(enter_points) # v in MY-TPAMI-20
        candidates = [] # C in MY-TPAMI-20
        currently_found_nearest_neighbors = set(enter_points) # W in MY-TPAMI-20

        # set variable for heapsort ordering, it depends on the direction of the trend score
        if not self._distance_algorithm.is_spatial():
            queue_multiplier = 1 # similarity metric
        else:
            queue_multiplier = -1 # distance metric

        # and initialize the priority queue with the existing candidates (from enter_points)
        for candidate in set(enter_points):
            distance = candidate.calculate_similarity(query_node)
            heapq.heappush(candidates, (distance*queue_multiplier, candidate))

        logging.info(f"Performing a k-NN search in layer {layer} ...")
        logging.debug(f"Candidates list: {candidates}")

        while len(candidates) > 0:
            logging.debug(f"Current NN found: {currently_found_nearest_neighbors}")
            # get the closest and furthest nodes from our candidates list
            furthest_node = self.find_furthest_element(query_node, currently_found_nearest_neighbors)
            logging.debug(f"Furthest node: {furthest_node}")
            _, closest_node = heapq.heappop(candidates)
            logging.debug(f" Closest node: {closest_node}")

            # closest node @candidates list is closer than furthest node @currently_found_nearest_neighbors            
            n2_is_closer_n1, _, _ = query_node.n2_closer_than_n1(n1=closest_node, n2=furthest_node)
            if n2_is_closer_n1:
                logging.debug("All elements in current nearest neighbors evaluated, exiting loop ...")
                break
            
            # get neighbor list in this layer
            _neighbor_list = closest_node.get_neighbors_at_layer(layer)
            logging.debug(f"Neighbour list of closest node: {_neighbor_list}")

            for neighbor in _neighbor_list:
                if neighbor not in visited_elements:
                    visited_elements.add(neighbor)
                    furthest_node = self.find_furthest_element(query_node, currently_found_nearest_neighbors)
                    
                    logging.debug(f"Neighbor: {neighbor}; furthest node: {furthest_node}")
                    # If the distance is smaller than the furthest node we have in our list, replace it in our list
                    n2_is_closer_n1, _, distance = query_node.n2_closer_than_n1(n2=neighbor, n1=furthest_node)
                    if n2_is_closer_n1 or len(currently_found_nearest_neighbors) < ef:
                        heapq.heappush(candidates, (distance*queue_multiplier, neighbor))
                        currently_found_nearest_neighbors.add(neighbor)
                        if len(currently_found_nearest_neighbors) > ef:
                            currently_found_nearest_neighbors.remove(self.find_furthest_element(query_node, currently_found_nearest_neighbors))
        logging.info(f"Current nearest neighbors at L{layer}: {currently_found_nearest_neighbors}")
        return currently_found_nearest_neighbors

    #TODO Check algorithm
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

    # Algorithm 4 in MY-TPAMI-20
    def select_neighbors_heuristics(self, new_node, candidates: set, M, 
                                    layer, extend_candidates, keep_pruned_conns):

        logging.info(f"Performing a k-NN heuristic search in layer {layer} ...")
        
        _r = set()
        _working_candidates = candidates
        if extend_candidates:
            logging.debug("Initial candidate list: {candidates}")
            logging.debug("Extending candidates ...")
            for candidate in candidates:
                _neighborhood_e = candidate.get_neighbors_at_layer(layer)
                for _neighbor in _neighborhood_e:
                    _working_candidates.add(_neighbor)

        logging.debug(f"Candidates list: {candidates}")
        
        _discarded = set()
        while len(_working_candidates) > 0 and len(_r) < M:
            # get nearest from W and from R and compare which is closer to new_node
            _elm_nearest_W  = self.find_nearest_element(new_node, _working_candidates)
            _working_candidates.remove(_elm_nearest_W)
            if len(_r) == 0: # trick for first iteration
                _r.add(_elm_nearest_W)
                logging.debug(f"Adding {_elm_nearest_W} to R")
                continue

            _elm_nearest_R  = self.find_nearest_element(new_node, _r)
            logging.debug(f"Nearest_R vs nearest_W: {_elm_nearest_R} vs {_elm_nearest_W}")
            n2_is_closer_n1, _, _ = new_node.n2_closer_than_n1(n1=_elm_nearest_R, n2=_elm_nearest_W)
            if n2_is_closer_n1:
                _r.add(_elm_nearest_W)
                logging.debug(f"Adding {_elm_nearest_W} to R")
            else:
                _discarded.add(_elm_nearest_W)
                logging.debug(f"Adding {_elm_nearest_W} to discarded set")

        if keep_pruned_conns:
            logging.debug("Keeping pruned connections ...")
            while len(_discarded) > 0 and len(_r) < M:
                _elm = self.find_nearest_element(new_node, _discarded)
                _discarded.remove(_elm)
                
                _r.add(_elm)
                logging.debug(f"Adding {_elm} to R")

        logging.debug(f"Neighbors: {_r}")
        return _r

    # Algorithm 3 in MY-TPAMI-20
    def select_neighbors_simple(self, new_node, candidates, M):
        """
        Get the M nearest neighbors.
        """
        nearest_neighbors = sorted(candidates, key=lambda obj: obj.calculate_similarity(new_node))
        logging.info(f"Neighbors to <{new_node}>: {nearest_neighbors}")
        if not self._distance_algorithm.is_spatial(): # similarity metric
            return nearest_neighbors[-M:] 
        else: # distance metric
            return nearest_neighbors[:M] 
    
    def select_neighbors(self, new_node, candidates, M, layer): # heuristic params
        if not self._heuristic:
            return self.select_neighbors_simple(new_node, candidates, M)
        else:
            return self.select_neighbors_heuristics(new_node, candidates, M,
                                                layer,
                                                self._extend_candidates, self._keep_pruned_conns)

    
    def find_furthest_element(self, node, nodes):
        if not self._distance_algorithm.is_spatial(): # similarity metric
            return min((n for n in nodes), key=lambda n: node.calculate_similarity(n), default=None)
        else: # distance metric
            return max((n for n in nodes), key=lambda n: node.calculate_similarity(n), default=None)

    def find_nearest_element(self, node, nodes):
        if not self._distance_algorithm.is_spatial(): # similarity metric
            return max((n for n in nodes), key=lambda n: node.calculate_similarity(n), default=None)
        else: # distance metric
            return min((n for n in nodes), key=lambda n: node.calculate_similarity(n), default=None)

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
        
        # check if the HNSW distance algorithm is the same as the one associated to the node
        # raise exception otherwise
        #TODO

        # update ef to efConstruction, if necessary
        if ef == 0: 
            ef = self._ef

        enter_point = self._descend_to_layer(query, layer=1) 
            
        # and now get the nearest elements
        current_nearest_elements = self.search_layer_knn(query, [enter_point], ef, 0)
        return self.select_neighbors(query, current_nearest_elements, k, 0)

    #TODO Check algorithm
    def percentage_search(self, query, percentage):
        """
            Performs a percentage search tºo retrieve nodes that satisfy a certain similarity threshold using the HNSW algorithm.
        
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
    
# unit test
from datalayer.node_hash import HashNode
from datalayer.tlsh_algorithm import TLSHHashAlgorithm
if __name__ == "__main__":
    # Create an HNSW structure
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.WARNING)
    myHNSW = HNSW(M=4, ef=4, Mmax=8, Mmax0=16, heuristic=False, distance_algorithm=TLSHHashAlgorithm)

    # Create the nodes based on TLSH Fuzzy Hashes
    node1 = HashNode("T12B81E2134758C0E3CA097B381202C62AC793B46686CD9E2E8F9190EC89C537B5E7AF4C", TLSHHashAlgorithm)
    node2 = HashNode("T10381E956C26225F2DAD9D5C2C5C1A337FAF3708A25012B8A1EACDAC00B37D557E0E714", TLSHHashAlgorithm)
    node3 = HashNode("T1DF8174A9C2A506F9C6FFC292D6816333FEF1B845C419121A0F91CF5359B5B21FA3A304", TLSHHashAlgorithm)
    node4 = HashNode("T1DF8174A9C2A506F9C6FFC292D6816333FEF1B845C419121A0F91CF5359B5B21FA3A304", TLSHHashAlgorithm)
    node5 = HashNode("T1DF8174A9C2A506F9C6FFC292D6816333FEF1B845C419121A0F91CF5359B5B21FA3A305", TLSHHashAlgorithm)
    nodes = [node1, node2, node3]

    # Insert nodes on the HNSW structure
    myHNSW.add_node(node1)
    myHNSW.add_node(node2)
    myHNSW.add_node(node3)
    try:
        myHNSW.add_node(node4)
    except NodeAlreadyExists:
        print(f"Node \"{node4.get_id()}\" cannot be inserted, already exists!")

    #breakpoint()
    print(f"Enter point: {myHNSW.get_enter_point()}")

    try:
        myHNSW.delete_node(node5)
    except NodeNotFound:
        print(f"Node \"{node5.get_id()}\" not found!")
    
    #myHNSW.delete_node(node3)

    # Perform k-nearest neighbor search based on TLSH fuzzy hash similarity
    query_node = HashNode("T1BF81A292E336D1F68224D4A4C751A2B3BB353CA9C2103BA69FA4C7908761B50F22E301", TLSHHashAlgorithm)
    for node in nodes:
        print(node, "Similarity score: ", node.calculate_similarity(query_node))

    print('Testing knn-search ...')
    results = myHNSW.knn_search(query_node, k=2, ef=4)
    print("Total neighbors found: ", len(results))
    for idx, node in enumerate(results):
        print(f"Node ID {idx + 1}: \"{node.get_id()}\"")

    # Perform percentage search to retrieve nodes above a similarity threshold
    #results = myHNSW.percentage_search(query_node, percentage=60)
    #print(results)

    # Dump created HNSW structure to disk
    myHNSW.dump("myHNSW.txt")

    # Restore HNSW structure from disk
    myHNSW = HNSW.load("myHNSW.txt")
