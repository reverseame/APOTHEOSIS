import numpy as np
import random
import pickle
import time
import logging
import heapq
import os
# for drawing
import networkx as nx
import matplotlib.pyplot as plt

# custom exceptions
from datalayer.errors import NodeNotFoundError
from datalayer.errors import NodeAlreadyExistsError
from datalayer.errors import HNSWUnmatchDistanceAlgorithmError
from datalayer.errors import HNSWUndefinedError
from datalayer.errors import HNSWIsEmptyError
from datalayer.errors import HNSWLayerDoesNotExistError
from datalayer.errors import HNSWEmptyLayerError

__author__ = "Daniel Huici Meseguer and Ricardo J. Rodríguez"
__copyright__ = "Copyright 2024"
__credits__ = ["Daniel Huici Meseguer", "Ricardo J. Rodríguez"]
__license__ = "GPL"
__version__ = "0.2"
__maintainer__ = "Daniel Huici"
__email__ = "reverseame@unizar.es"
__status__ = "Development"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.getLogger('pickle').setLevel(logging.WARNING)
logging.getLogger('numpy').setLevel(logging.WARNING)
logging.getLogger('time').setLevel(logging.WARNING)
logging.getLogger('networkx').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

class HNSW:
    # initial layer will have index 0
    
    def __init__(self, M, ef, Mmax, Mmax0,
                    distance_algorithm=None,
                    heuristic=False, extend_candidates=True, keep_pruned_conns=True):
        """Default constructor."""
        # HNSW parameters
        self._M = M
        self._Mmax = Mmax # max links per node
        self._Mmax0 = Mmax0 # max links per node at layer 0 
        self._ef = ef
        self._mL = 1.0 / np.log(self._M)
        self._enter_point = None
        # keep the associated distance algorithm and set self._queue_factor appropriately
        self._distance_algorithm = distance_algorithm
        self._set_queue_factor()
        # dictionary to store all nodes of the HNSW, per level
        self._nodes = dict()
        # select neighbors heuristic params
        self._heuristic = heuristic
        self._extend_candidates = extend_candidates
        self._keep_pruned_conns = keep_pruned_conns
    
    def _is_empty(self):
        """Returns True if the HNSW structure contains no node, False otherwise."""
        return (self._enter_point is None)
   
    def _assert_no_empty(self):
        """Raises HNSWIsEmptyError if the HNSW structure is empty."""
        if self._is_empty():
            raise HNSWIsEmptyError

    def get_enter_point(self):
        """Getter for _enter_point."""
        return self._enter_point
    
    def get_distance_algorithm(self):
        """Getter for _distance_algorithm."""
        return self._distance_algorithm
    
    def get_queue_factor(self):
        """Getter for _queue_factor."""
        return self._queue_factor
    
    def get_M(self):
        """Getter for _M."""
        return self._M
    
    def get_Mmax(self):
        """Getter for _Mmax."""
        return self._Mmax
    
    def get_Mmax0(self):
        """Getter for _Mmax0."""
        return self._Mmax0
    
    def get_ef(self):
        """Getter for _ef."""
        return self._ef

    def _insert_node(self, node):
        """Inserts node in the dict of the HNSW structure.

        Arguments:
        node -- the new node to insert
        """
        _layer = node.get_max_layer()
        if self._nodes.get(_layer) is None:
            self._nodes[_layer] = list()
        
        self._nodes[_layer].append(node)

    def _set_queue_factor(self):
        if not self._distance_algorithm.is_spatial():
            self._queue_factor = 1 # similarity metric
        else:
            self._queue_factor = -1 # distance metric

    def _descend_to_layer(self, query_node, layer=0):
        """Goes down to a specific layer and returns the enter point of that layer, 
        which is the nearest element to query_node.
        
        Arguments:
        query_node  -- the node to be inserted
        layer       -- the target layer (default 0)
        """
        enter_point = self._enter_point
        for layer in range(self._enter_point.get_max_layer(), layer, -1): # Descend to the given layer
            logger.debug(f"Visiting layer {layer}, ep: {enter_point}")
            current_nearest_elements = self._search_layer_knn(query_node, [enter_point], 1, layer)
            logger.debug(f"Current nearest elements: {current_nearest_elements}")
            if len(current_nearest_elements) > 0:
                if enter_point.get_id() != query_node.get_id():
                    # get the nearest element to query node if the enter_point is not the query node itself
                    enter_point = self._find_nearest_element(query_node, current_nearest_elements)
            else: #XXX is this path even feasible?
                logger.warning("No closest neighbor found at layer {}".format(layer))

        return enter_point

    def _assert_same_distance_algorithm(self, node):
        """Checks if the distance algorithm associated to node matches with the distance algorithm
        associated to the HNSW structure and raises HNSWUnmatchDistanceAlgorithmError when they do not match
        
        Arguments:
        node    -- the node to check
        """
        if node.get_distance_algorithm() != self.get_distance_algorithm():
            raise HNSWUnmatchDistanceAlgorithmError

    def add_node(self, new_node):
        """Adds a new node to the HNSW structure. On success, it return True
        Raises HNSWUnmatchDistanceAlgorithmError if the distance algorithm of the new node is distinct than 
        the distance algorithm associated to the HNSW structure.
        Raises NodeAlreadyExistsError if the HNSW already contains a node with the same ID as the new node.
        
        Arguments:
        new_node    -- the node to be added
        """
        # check if the HNSW distance algorithm is the same as the one associated to the node
        self._assert_same_distance_algorithm(new_node)
        
        enter_point = self._enter_point
        # Calculate the layer to which the new node belongs
        new_node_layer = int(-np.log(random.uniform(0,1)) * self._mL) // 1 # l in MY-TPAMI-20
        new_node.set_max_layer(new_node_layer)
        logger.info(f"New node to insert: \"{new_node.get_id()}\" (assigned level: {new_node_layer})")
        
        if enter_point is not None:
            # checks if the enter point matches the new node and raises exception
            if enter_point.get_id() == new_node.get_id():
                raise NodeAlreadyExistsError
            
            Lep = enter_point.get_max_layer()
           
            logger.debug(f"Descending to layer {new_node_layer}")
            # Descend from the entry point to the layer of the new node...
            enter_point = self._descend_to_layer(new_node, layer=new_node_layer)

            logger.debug(f"Inserting \"{new_node.get_id()}\" using \"{enter_point}\" as enter point ...")
            # Insert the new node
            self._insert_node_to_layers(new_node, [enter_point])

            # Update enter_point of the HNSW, if necessary
            if new_node_layer > Lep:
                self._enter_point = new_node
                logger.info(f"Setting \"{new_node.get_id()}\" as enter point ... ")

        else:
            self._enter_point = new_node
            logger.info(f"Updating \"{new_node.get_id()}\" as enter point ... ")
        
        # store it now in its corresponding level
        self._insert_node(new_node)
        return True

    def _get_random_node_at_layer(self, layer, exclude_nodes: list=None):
        """Returns a random node at a given layer, excluding the nodes in exclude_nodes from being selected
        It may raise the following exceptions:
            * HNSWLayerDoesNotExistError if the layer does not exist and 
            * HNSWEmptyLayerError if the layer contains no possible nodes for selection

        Arguments:
        layer           -- layer number
        exclude_nodes   -- set of nodes to exclude from selection set
        """

        if self._nodes.get(layer) is None:
            raise HNSWLayerDoesNotExistError
        
        if len(self._nodes[layer]) == 0:
            raise HNSWEmptyLayerError
       
        _candidates_set = set(self._nodes[layer])
        if exclude_nodes is not None:
            logger.debug(f"Excluding nodes from random selection at L{layer}: <{[node.get_id() for node in exclude_nodes]}>")
            _candidates_set = _candidates_set - set(exclude_nodes)

        _elm = random.choice(list(_candidates_set))
        logger.debug(f"Random node chosen at L{layer}: \"{_elm.get_id()}\"")
        return _elm

    def _delete_neighbors_connections(self, node):
        """Given a node, delete the connections to their neighbors.

        Arguments:
        node    -- the node to delete
        """

        logger.debug(f"Deleting neighbors of \"{node.get_id()}\"")
        for layer in range(node.get_max_layer() + 1):
            for neighbor in node.get_neighbors_at_layer(layer):
                logger.debug(f"Deleting at L{layer} link \"{neighbor.get_id()}\"")
                neighbor.remove_neighbor(layer, node)

    def _delete_node(self, node):
        """Delete a node from the dict of the HNSW structure.
        It raises HNSWUndefinedError if the node to delete was not stored in the dict 

        Arguments:
        node    -- the node to delete
        """
        _layer = node.get_max_layer()
        try:
            self._nodes[_layer].remove(node)
            if len(self._nodes[_layer]) == 0: # remove this key, it is empty
                self._nodes.pop(_layer)
        except:
            raise HNSWUndefinedError

    def delete_node(self, node):
        """Deletes a node of the HNSW structure. On success, it returns True
        It may raise several exceptions:
            * HNSWIsEmptyError when the HNSW structure has no nodes.
            * NodeNotFoundError when the node to delete is not found in the HNSW structure.
            * HNSWUndefinedError when no neighbor is found at layer 0 (shall never happen this!).
            * HNSWUnmatchDistanceAlgorithmError when the distance algorithm of the node to delete
              does not match the distance algorithm associated to the HNSW structure.
        
        Arguments:
        node    -- the node to delete
        """
        # check if the HNSW distance algorithm is the same as the one associated to the node to delete
        self._assert_same_distance_algorithm(node)
        # check if it is empty
        self._assert_no_empty()
        
        # OK, you can try to search and delete the given node now
        # from the enter_point, reach the node, if exists
        enter_point = self._descend_to_layer(node)
        # now checks for the node, if it is in this layer
        found_node = self._search_layer_knn(node, [enter_point], 1, 0)
        if len(found_node) == 1:
            found_node = found_node.pop()
            if found_node.get_id() == node.get_id():
                logger.info(f"Node \"{node.get_id()}\" found! Deleting it ...")
                if found_node == self._enter_point: # cover the case we try to delete enter point
                    logger.info("The node to delete is the enter point! Searching for a new enter point ...")
                    # iterate layers until we find a neighbor
                    for layer in range(self._enter_point.get_max_layer(), -1, -1):
                        _neighs_list = found_node.get_neighbors_at_layer(layer)
                        if len(_neighs_list) == 0:
                            if layer == 0: # neighbors list is empty and we are at layer 0... check dict nodes
                                if self._nodes.get(layer) is None or len(self._nodes[layer]) == 1: # the structure will be now empty
                                        # it may happen the node is not at layer 0, as we only keep them in the max layer
                                    logger.debug("No enter point! HNSW is now empty")
                                    self._enter_point = None
                                else:
                                    # it may happen we have other nodes in this layer, but not connected to found_node
                                    # if so, select one of them randomly
                                    self._enter_point = self._get_random_node_at_layer(layer, exclude_node=found_node)
                                    logger.debug(f"New enter point randomly selected: \"{self._enter_point.get_id()}\"")

                            continue # this layer is empty, continue until we find one layer with neighbors
                        
                        closest_neighbor = self._select_neighbors(found_node, _neighs_list, M=1, layer=layer)
                        if len(closest_neighbor) == 1: # select this as new enter point and exit the loop
                            self._enter_point = closest_neighbor.pop()
                            break

                # now safely delete neighbor's connections
                self._delete_neighbors_connections(found_node)
                self._delete_node(found_node)
            else:
                logger.info(f"Node \"{node.get_id()}\" not found at layer 0 ...")
                raise NodeNotFoundError
        else:
            logger.info(f"No nearest neighbor found at layer 0. HNSW empty?")
            # It should always get one nearest neighbor, unless it is empty
            raise HNSWIsEmptyError
        return True

    def _already_exists(self, query_node, node_list) -> bool:
        """Returns True if query_node is contained in node_list, False otherwise.

        Arguments:
        query_node  -- the node to search
        node_list   -- the list of nodes where to search
        """
        for node in node_list:
            if node.get_id() == query_node.get_id():
                return True
        return False
    
    def _shrink_nodes(self, nodes, layer):
        """Shrinks the maximum number of neighbors of nodes in a given layer.
        The maximum value depends on the layer (MMax0 for layer 0 or Mmax for other layers).

        Arguments:
        nodes   -- list of nodes to shrink
        layer   -- current layer to search neighbors and update in each node 
        """

        mmax = self._Mmax0 if layer == 0 else self._Mmax
        for _node in nodes:
            _list = _node.get_neighbors_at_layer(layer)
            if (len(_list) > mmax):
                _shrinked_neighbors = self._select_neighbors(_node, _list, mmax, layer)
                _node.set_neighbors_at_layer(layer, set(_shrinked_neighbors))
                logger.debug(f"Node {_node.get_id()} exceeded Mmax. New neighbors: {[n.get_id() for n in _node.get_neighbors_at_layer(layer)]}")

    def _insert_node_to_layers(self, new_node, enter_point):
        """Inserts the new node from the minimum layer between HNSW enter point and the new node until layer 0.
        The first visited layer uses enter point as initial point for the best place to insert.
        It raises NodeAlreadyExistsError if the node already exists in the HNSW structure.

        Arguments:
        new_node    -- the node to insert
        enter_point -- the enter point to the first layer to visit 
        """
        
        min_layer = min(self._enter_point.get_max_layer(), new_node.get_max_layer())
        for layer in range(min_layer, -1, -1):
            currently_found_nn = self._search_layer_knn(new_node, enter_point, self._ef, layer)
            new_neighbors = self._select_neighbors(new_node, currently_found_nn, self._M, layer)
            logger.debug(f"Found nn at L{layer}: {currently_found_nn}")

            if self._already_exists(new_node, currently_found_nn) or \
                    self._already_exists(new_node, new_neighbors):
                max_layer = new_node.get_max_layer()
                if max_layer > layer: # if the previous node is found but in a lower layer than the assigned to the new node
                    for _layer in range(layer + 1, max_layer + 1): # delete all links set with the new node in upper layers
                        for neighbor in new_node.get_neighbors_at_layer(_layer):
                            neighbor.remove_neighbor(_layer, new_node)
                raise NodeAlreadyExistsError

            # connect both nodes bidirectionally
            for neighbor in new_neighbors: 
                neighbor.add_neighbor(layer, new_node)
                new_node.add_neighbor(layer, neighbor)
                logger.debug(f"Connections added at L{layer} between {new_node} and {neighbor}")
            
            # shrink (when we have exceeded the Mmax limit)
            self._shrink_nodes(new_neighbors, layer)
            enter_point.extend(currently_found_nn)
        
    def _search_layer_knn(self, query_node, enter_points, ef, layer):
        """Performs a k-NN search in a specific layer of the graph.

        Arguments:
        query_node      -- the node to search
        enter_points    -- current enter points
        ef              -- number of nearest elements to query_node to return
        layer           -- layer number
        """
        visited_elements = set(enter_points) # v in MY-TPAMI-20
        candidates = [] # C in MY-TPAMI-20
        currently_found_nearest_neighbors = set(enter_points) # W in MY-TPAMI-20

        # get variable for heapsort ordering, it depends on the direction of the trend score
        queue_factor = self.get_queue_factor()

        # and initialize the priority queue with the existing candidates (from enter_points)
        for candidate in set(enter_points):
            distance = candidate.calculate_similarity(query_node)
            heapq.heappush(candidates, (distance*queue_factor, candidate))

        logger.debug(f"Performing a k-NN search of \"{query_node.get_id()}\" in layer {layer} ...")
        logger.debug(f"Candidates list: {candidates}")

        while len(candidates) > 0:
            logger.debug(f"Current NN found: {currently_found_nearest_neighbors}")
            # get the closest and furthest nodes from our candidates list
            furthest_node = self._find_furthest_element(query_node, currently_found_nearest_neighbors)
            logger.debug(f"Furthest node: {furthest_node}")
            _, closest_node = heapq.heappop(candidates)
            logger.debug(f" Closest node: {closest_node}")

            # closest node @candidates list is closer than furthest node @currently_found_nearest_neighbors            
            n2_is_closer_n1, _, _ = query_node.n2_closer_than_n1(n1=closest_node, n2=furthest_node)
            if n2_is_closer_n1:
                logger.debug("All elements in current nearest neighbors evaluated, exiting loop ...")
                break
            
            # get neighbor list in this layer
            _neighbor_list = closest_node.get_neighbors_at_layer(layer)
            logger.debug(f"Neighbor list of closest node: {_neighbor_list}")

            for neighbor in _neighbor_list:
                if neighbor not in visited_elements:
                    visited_elements.add(neighbor)
                    furthest_node = self._find_furthest_element(query_node, currently_found_nearest_neighbors)
                    
                    logger.debug(f"Neighbor: {neighbor}; furthest node: {furthest_node}")
                    # If the distance is smaller than the furthest node we have in our list, replace it in our list
                    n2_is_closer_n1, _, distance = query_node.n2_closer_than_n1(n2=neighbor, n1=furthest_node)
                    if n2_is_closer_n1 or len(currently_found_nearest_neighbors) < ef:
                        heapq.heappush(candidates, (distance*queue_factor, neighbor))
                        currently_found_nearest_neighbors.add(neighbor)
                        if len(currently_found_nearest_neighbors) > ef:
                            currently_found_nearest_neighbors.remove(self._find_furthest_element(query_node, currently_found_nearest_neighbors))
        logger.debug(f"Current nearest neighbors at L{layer}: {currently_found_nearest_neighbors}")
        return currently_found_nearest_neighbors

    def _search_layer_threshold(self, query_node, enter_points, threshold, n_hops, layer):
        """Performs a threshold search at a given layer of the graph.

        Arguments:
        query_node      -- the node to search
        enter_points    -- current enter points
        threshold       -- threshold similarity
        n_hops          -- number of hops to perfom from each nearest neighbor
        layer           -- layer number
        """
        visited_elements = set(enter_points)
        candidates = []
        final_elements = set() 

        # get variable for heapsort ordering, it depends on the direction of the trend score
        queue_factor = self.get_queue_factor()
        
        # initialize the priority queue with the existing candidates
        for candidate in enter_points:
            satisfies_treshold, distance = query_node.n1_above_threshold(n1=candidate, threshold=threshold)
            heapq.heappush(candidates, (distance*queue_factor, candidate))
            if (satisfies_treshold): # select this node for the final set
                final_elements.add(candidate)

        while len(candidates) > 0 and n_hops > 0:
            # get the closest node from our candidates list
            _, closest_node = heapq.heappop(candidates)

            # add new candidates to the priority queue
            for neighbor in closest_node.get_neighbors_at_layer(layer):
                if neighbor not in visited_elements:
                    visited_elements.add(neighbor)
                    satisfies_treshold, distance = query_node.n1_above_threshold(neighbor, threshold)
                    heapq.heappush(candidates, (distance*queue_factor, neighbor))
                    if (satisfies_treshold): # select this node for the final set
                        final_elements.add(neighbor)
            n_hops -= 1

        return final_elements

    def _select_neighbors_heuristics(self, node, candidates: set, M, 
                                    layer, extend_candidates, keep_pruned_conns):
        """Returns the M nearest neighbors to node from the list of candidates.
        This corresponds to Algorithm 4 in MY-TPAMI-20.

        Arguments:
        node                -- base element
        candidates          -- candidate set
        M                   -- number of neighbors to return
        layer               -- layer number
        extend_candidates   -- flag to indicate whether or not to extend candidate list
        keep_pruned_conns   -- flag to indicate whether or not to add discarded elements
        """

        logger.debug(f"Selecting neighbors with a heuristic search in layer {layer} ...")
        
        _r = set()
        _working_candidates = candidates
        if extend_candidates:
            logger.debug(f"Initial candidate set: {candidates}")
            logger.debug("Extending candidates ...")
            for candidate in candidates:
                _neighborhood_e = candidate.get_neighbors_at_layer(layer)
                for _neighbor in _neighborhood_e:
                    _working_candidates.add(_neighbor)

        logger.debug(f"Candidates list: {candidates}")
        
        _discarded = set()
        while len(_working_candidates) > 0 and len(_r) < M:
            # get nearest from W and from R and compare which is closer to new_node
            _elm_nearest_W  = self._find_nearest_element(node, _working_candidates)
            _working_candidates.remove(_elm_nearest_W)
            if len(_r) == 0: # trick for first iteration
                _r.add(_elm_nearest_W)
                logger.debug(f"Adding {_elm_nearest_W} to R")
                continue

            _elm_nearest_R  = self._find_nearest_element(node, _r)
            logger.debug(f"Nearest_R vs nearest_W: {_elm_nearest_R} vs {_elm_nearest_W}")
            n2_is_closer_n1, _, _ = node.n2_closer_than_n1(n1=_elm_nearest_R, n2=_elm_nearest_W)
            if n2_is_closer_n1:
                _r.add(_elm_nearest_W)
                logger.debug(f"Adding {_elm_nearest_W} to R")
            else:
                _discarded.add(_elm_nearest_W)
                logger.debug(f"Adding {_elm_nearest_W} to discarded set")

        if keep_pruned_conns:
            logger.debug("Keeping pruned connections ...")
            while len(_discarded) > 0 and len(_r) < M:
                _elm = self._find_nearest_element(node, _discarded)
                _discarded.remove(_elm)
                
                _r.add(_elm)
                logger.debug(f"Adding {_elm} to R")

        logger.debug(f"Neighbors: {_r}")
        return _r

    def _select_neighbors_simple(self, node, candidates: set, M):
        """Returns the M nearest neighbors to node from the list of candidates.
        This corresponds to Algorithm 3 in MY-TPAMI-20.

        Arguments:
        node        -- base element
        candidates  -- candidate set
        M           -- number of neighbors to return
        """
        nearest_neighbors = sorted(candidates, key=lambda obj: obj.calculate_similarity(node))
        logger.debug(f"Neighbors to <{node}>: {nearest_neighbors}")
        if not self._distance_algorithm.is_spatial(): # similarity metric
            return nearest_neighbors[-M:] 
        else: # distance metric
            return nearest_neighbors[:M] 
    
    def _select_neighbors(self, node, candidates, M, layer): # heuristic params
        """Returns the M nearest neighbors to node from the set of candidates.
        If not _heuristic, it uses a simple selection of neighbors (Algorithm 3 in MY-TPAMI-20).
        Otherwise, it uses a heuristic selection (Algorithm 4 in MY-TPAMI-20)

        Arguments:
        node        -- base element
        candidates  -- candidate set
        M           -- number of neighbors to return
        layer       -- layer number
        """
        if not self._heuristic:
            return self._select_neighbors_simple(node, candidates, M)
        else:
            return self._select_neighbors_heuristics(node, candidates, M,
                                                layer,
                                                self._extend_candidates, self._keep_pruned_conns)
    
    def _find_furthest_element(self, node, nodes):
        """Returns the furthest element from nodes to node.

        Arguments:
        node    -- the base node
        nodes   -- the list of candidate nodes 
        """
        if not self._distance_algorithm.is_spatial(): # similarity metric
            return min((n for n in nodes), key=lambda n: node.calculate_similarity(n), default=None)
        else: # distance metric
            return max((n for n in nodes), key=lambda n: node.calculate_similarity(n), default=None)

    def _find_nearest_element(self, node, nodes):
        """Returns the nearest element from nodes to node.

        Arguments:
        node    -- the base node
        nodes   -- the list of candidate nodes 
        """
        if not self._distance_algorithm.is_spatial(): # similarity metric
            return max((n for n in nodes), key=lambda n: node.calculate_similarity(n), default=None)
        else: # distance metric
            return min((n for n in nodes), key=lambda n: node.calculate_similarity(n), default=None)

    def dump(self, file):
        """Saves HNSW structure to permanent storage.

        Arguments:
        file    -- filename to save 
        """

        with open(file, "wb") as f:
            pickle.dump(self, f, protocol=pickle.DEFAULT_PROTOCOL)

    @classmethod
    def load(cls, file):
        """Restores HNSW structure from permanent storage.
        
        Arguments:
        file    -- filename to load
        """
        with open(file, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected an instance of {cls.__name__}, but got {type(obj).__name__}")
        return obj

    def _node_list_to_dict(self, query, node_list: list) -> dict:
        """Returns a dictionary of nodes where the key is the similarity score with the query node and the values are
        the corresponding nodes.

        Arguments:
        query       -- the base node
        node_list   -- the list of nodes to transform in dict
        """
        _result = {}
        for _node in node_list:
            _value = _node.calculate_similarity(query)
            if _result.get(_value) is None:
                _result[_value] = []
            _result[_value].append(_node)
        return _result

    def _expand_with_neighbors(self, _nodes, _layer=0):
        """Expands the set of nodes with their neighbors
        
        Arguments:
        _nodes  -- set of nodes to expand
        _layer  -- layer level (default 0)
        """
        _result = set()
        for _node in _nodes:
            _result.add(_node)
            for _neighbor in _node.get_neighbors_at_layer(_layer):
                _result.add(_neighbor)
        return _result

    def knn_search(self, query, k, ef=0): 
        """Performs k-nearest neighbors search using the HNSW structure.
        It returns a dictionary (keys are similarity score) of k nearest neighbors (the values inside the dict) to the query node.
        It raises the following exceptions:
            * HNSWUnmatchDistanceAlgorithmError if the distance algorithm of the new node is distinct than 
              the distance algorithm associated to the HNSW structure.
            * HNSWIsEmptyError if the HNSW structure is empty
        
        Arguments:
        query   -- the node for which to find the k nearest neighbors
        k       -- the number of nearest neighbors to retrieve
        ef      -- the exploration factor (controls the search recall)
        """
        
        # check if the HNSW distance algorithm is the same as the one associated to the query node
        self._assert_same_distance_algorithm(query)
        # check if the HNSW is empty
        self._assert_no_empty()

        # update ef to efConstruction, if necessary
        if ef == 0: 
            ef = self._ef

        logger.info(f"Performing a KNN search of \"{query.get_id()}\" with ef={ef} ...")
        enter_point = self._descend_to_layer(query, layer=1) 
            
        # and now get the nearest elements
        current_nearest_elements = self._search_layer_knn(query, [enter_point], ef, 0)
        current_nearest_elements = self._expand_with_neighbors(current_nearest_elements)
        _knn_list = self._select_neighbors(query, current_nearest_elements, k, 0)
        _knn_list = sorted(_knn_list, key=lambda obj: obj.calculate_similarity(query))
        logger.info(f"KNNs found (sorted list): {_knn_list} ...")
        
        # return a dictionary of nodes and similarity score
        return self._node_list_to_dict(query, _knn_list)

    def threshold_search(self, query, threshold, n_hops):
        """Performs a threshold search to retrieve nodes that satisfy a certain similarity threshold using the HNSW structure.
        It returns a list of nearest neighbor nodes to query that satisfy the specified similarity threshold.
        It raises the following exceptions:
            * HNSWUnmatchDistanceAlgorithmError if the distance algorithm of the new node is distinct than 
              the distance algorithm associated to the HNSW structure.
            * HNSWIsEmptyError if the HNSW structure is empty

        Arguments:
        query      -- the query node for which to find the neighbors with a similarity above the given percentage
        threshold  -- the similarity threshold to satisfy 
        n_hops     -- number of hops to perform from each nearest neighbor
        """

        # check if the HNSW distance algorithm is the same as the one associated to the query node
        self._assert_same_distance_algorithm(query)
        # check if the HNSW is empty
        self._assert_no_empty()
        
        ef = self._ef # exploration factor always set to default 

        logger.info(f"Performing a threshold search of \"{query.get_id()}\" with threshold {threshold} and nhops {n_hops} (ef={ef}) ...")
        # go to layer 0 (enter point is in layer 1)
        enter_point = self._descend_to_layer(query, layer=1)
        # get list of neighbors, considering threshold
        _current_neighs = self._search_layer_threshold(query, [enter_point], threshold, n_hops, layer=0)
        _current_neighs = sorted(_current_neighs, key=lambda obj: obj.calculate_similarity(query))
        logger.info(f"Neighbors found (sorted list): {_current_neighs} ...")

        # return a dictionary of nodes and similarity score
        return self._node_list_to_dict(query, _current_neighs)
    
    def _get_edge_labels(self, G: nx.Graph):
        """Returns the labels (distance score) of the edges in the graph G
        
        Arguments:
        G   -- graph from which the edge labels are retrieved
        """
        edge_labels = {}
        for edge in list(G.edges):
            try:
                edge_labels[edge] = G.get_edge_data(*edge)['label']
            except KeyError as e:
                edge_labels[edge] = 'None'
        
        return edge_labels

    def draw(self, filename: str, show_distance: bool=True, format="pdf"):
        """Creates a digraph figure per level and saves it to a filename file.

        Arguments:
        filename        -- filename to create (with extension)
        show_distance   -- to show the distance metric in the edges (default is True)
        format          -- file extension
        """
        
        # iterate on layers
        for _layer in sorted(self._nodes.keys(), reverse=True):
            G = nx.Graph()
            # iterate on nodes
            for _node in self._nodes[_layer]:
                _node_label = _node.get_id()[-5:]
                # iterate on neighbors
                for _neighbor in _node.get_neighbors_at_layer(_layer):
                    _neigh_label = _neighbor.get_id()[-5:]
                    _edge_label = ""
                    if show_distance:
                        _edge_label = _node.calculate_similarity(_neighbor)
                    # nodes are automatically created if they are not already in the graph
                    G.add_edge(_node_label, _neigh_label, label=_edge_label)
                    
            pos = nx.spring_layout(G, k=5)
            nx.draw(G, pos, node_size=1500, node_color='yellow', font_size=8, font_weight='bold', with_labels=True)
            nx.draw_networkx_edge_labels(G, pos, edge_labels = self._get_edge_labels(G), font_size=6)
            plt.savefig(f"L{_layer}" + filename, format=format)
            plt.clf()

# unit test
import argparse
from datalayer.node.node_hash import HashNode
from datalayer.hash_algorithm.tlsh_algorithm import TLSHHashAlgorithm

def print_results(results: dict, show_keys=False):
    # iterate now in the results. If we sort the keys, we can get them ordered by similarity score
    keys = sorted(results.keys())
    idx = 1
    for key in keys:
        for node in results[key]:
            _str = f"Node ID {idx}: \"{node.get_id()}\""
            if show_keys:
                _str += f" (score: {key})"
            print(_str)
            idx += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--M', type=int, default=4, help="Number of established connections of each node (default=4)")
    parser.add_argument('--ef', type=int, default=4, help="Exploration factor (determines the search recall, default=4)")
    parser.add_argument('--Mmax', type=int, default=8, help="Max links allowed per node at any layer, but layer 0 (default=8)")
    parser.add_argument('--Mmax0', type=int, default=16, help="Max links allowed per node at layer 0 (default=16)")
    parser.add_argument('--heuristic', help="Create a HNSW structure using a heuristic to select neighbors rather than a simple selection algorithm (disabled by default)", action='store_true')
    parser.add_argument('--no-extend-candidates', help="Neighbor heuristic selection extendCandidates parameter (enabled by default)", action='store_true')
    parser.add_argument('--no-keep-pruned-conns', help="Neighbor heuristic selection keepPrunedConns parameter (enabled by default)", action='store_true')
    # get log level from command line
    parser.add_argument('-log', '--loglevel', choices=["debug", "info", "warning", "error", "critical"], default='warning', help="Provide logging level (default=warning)")

    args = parser.parse_args()
    
    # Create an HNSW structure
    logging.basicConfig(format='%(levelname)s:%(message)s', level=args.loglevel.upper())
    logger.setLevel(args.loglevel.upper())

    myHNSW = HNSW(M=args.M, ef=args.ef, Mmax=args.Mmax, Mmax0=args.Mmax0,\
                    heuristic=args.heuristic, extend_candidates=not args.no_extend_candidates, keep_pruned_conns=not args.no_keep_pruned_conns,\
                    distance_algorithm=TLSHHashAlgorithm)

    # Create the nodes based on TLSH Fuzzy Hashes
    node1 = HashNode("T12B81E2134758C0E3CA097B381202C62AC793B46686CD9E2E8F9190EC89C537B5E7AF4C", TLSHHashAlgorithm)
    node2 = HashNode("T10381E956C26225F2DAD9D5C2C5C1A337FAF3708A25012B8A1EACDAC00B37D557E0E714", TLSHHashAlgorithm)
    node3 = HashNode("T1DF8174A9C2A506F9C6FFC292D6816333FEF1B845C419121A0F91CF5359B5B21FA3A304", TLSHHashAlgorithm)
    node4 = HashNode("T1DF8174A9C2A506F9C6FFC292D6816333FEF1B845C419121A0F91CF5359B5B21FA3A304", TLSHHashAlgorithm)
    node5 = HashNode("T1DF8174A9C2A506F9C6FFC292D6816333FEF1B845C419121A0F91CF5359B5B21FA3A305", TLSHHashAlgorithm)
    nodes = [node1, node2, node3]

    print("Testing add_node ...")
    # Insert nodes on the HNSW structure
    if myHNSW.add_node(node1):
        print(f"Node \"{node1.get_id()}\" inserted correctly.")
    if myHNSW.add_node(node2):
        print(f"Node \"{node2.get_id()}\" inserted correctly.")
    if myHNSW.add_node(node3):
        print(f"Node \"{node3.get_id()}\" inserted correctly.")
    try:
        myHNSW.add_node(node4)
        print(f"WRONG --> Node \"{node4.get_id()}\" inserted correctly.")
    except NodeAlreadyExistsError:
        print(f"Node \"{node4.get_id()}\" cannot be inserted, already exists!")

    print(f"Enter point: {myHNSW.get_enter_point()}")

    try:
        myHNSW.delete_node(node5)
    except NodeNotFoundError:
        print(f"Node \"{node5.get_id()}\" not found!")
   
    print("Testing delete_node ...")
    myHNSW.delete_node(node1)
    #myHNSW.delete_node(node2)
    #myHNSW.delete_node(node3)

    # Perform k-nearest neighbor search based on TLSH fuzzy hash similarity
    query_node = HashNode("T1BF81A292E336D1F68224D4A4C751A2B3BB353CA9C2103BA69FA4C7908761B50F22E301", TLSHHashAlgorithm)
    for node in nodes:
        print(node, "Similarity score: ", node.calculate_similarity(query_node))

    print('Testing knn_search ...')
    try:
        results = myHNSW.knn_search(query_node, k=2, ef=4)
        print("Total neighbors found: ", len(results))
        print_results(results)
    except HNSWIsEmptyError:
        print("ERROR: performing a KNN search in an empty HNSW")
        
    print('Testing threshold_search ...')
    # Perform threshold search to retrieve nodes above a similarity threshold
    try:
        results = myHNSW.threshold_search(query_node, threshold=220, n_hops=3)
        print_results(results, show_keys=True)
    except HNSWIsEmptyError:
        print("ERROR: performing a KNN search in an empty HNSW")

    # Draw it
    myHNSW.draw("unit_test.pdf")

    # Dump created HNSW structure to disk
    myHNSW.dump("myHNSW.txt")

    # Restore HNSW structure from disk
    myHNSW = HNSW.load("myHNSW.txt")

