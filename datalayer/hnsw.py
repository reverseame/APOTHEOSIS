# -*- coding: utf-8 -*-
import numpy as np
import random
import pickle
import time
import heapq
import os
import struct
import logging
logger = logging.getLogger(__name__)

# for drawing
import networkx as nx
import matplotlib.pyplot as plt

from common.constants import *
# custom exceptions
from common.errors import HNSWUnmatchDistanceAlgorithmError
from common.errors import HNSWUndefinedError
from common.errors import HNSWIsEmptyError
from common.errors import HNSWLayerDoesNotExistError
from common.errors import HNSWEmptyLayerError
from common.errors import ApotFileFormatUnsupportedError

# for compressed dumping
import tempfile
import gzip as gz 
import io

__author__ = "Daniel Huici Meseguer and Ricardo J. Rodríguez"
__copyright__ = "Copyright 2024"
__credits__ = ["Daniel Huici Meseguer", "Ricardo J. Rodríguez"]
__license__ = "GPL"
__version__ = "0.2"
__maintainer__ = "Daniel Huici"
__email__ = "reverseame@unizar.es"
__status__ = "Development"

logging.getLogger('pickle').setLevel(logging.WARNING)
logging.getLogger('numpy').setLevel(logging.WARNING)
logging.getLogger('time').setLevel(logging.WARNING)
logging.getLogger('networkx').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

class HNSW:
    # initial layer will have index 0
    
    def __init__(self, M, ef, Mmax, Mmax0,
                    distance_algorithm=None,
                    heuristic=False, extend_candidates=True, keep_pruned_conns=True,
                    beer_factor: float=0):
        """Default constructor."""
        # HNSW parameters
        self._M = M
        self._Mmax = Mmax # max links per node
        self._Mmax0 = Mmax0 # max links per node at layer 0 
        self._ef = ef
        mL = np.log(M)
        if mL != 0:
            self._mL = 1.0 / mL
        else:
            self._mL = 0
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
        # research stuff
        self._beer_factor = beer_factor # random walk factor [0, 1), defines the probability of doing a random walk

    def _is_empty(self):
        """Returns True if the HNSW structure contains no node, False otherwise."""
        return (self._enter_point is None)
   
    def _assert_no_empty(self):
        """Raises HNSWIsEmptyError if the HNSW structure is empty."""
        if self._is_empty():
            raise HNSWIsEmptyError
    
    def _assert_layer_exists(self, layer):
        """Raises HNSWLayerDoesNotExistError if the HNSW does not have any node at the given layer.."""
        if self._nodes.get(layer) is None:
            raise HNSWLayerDoesNotExistError
    
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
    
    def get_nodes_at_layer(self, layer=0):
        """Returns the nodes at the given layer of the HNSW structure.
        It may raise the following exceptions:
            * HNSWIsEmptyError if the HNSW is empty
            * HNSWLayerDoesNotExistError if the layer does not exist

        Arguments:
        layer   -- layer number
        """

        self._assert_no_empty()
        self._assert_layer_exists(layer)

        return self._nodes[layer]

    def _insert_node(self, node):
        """Inserts node in the dict of the HNSW structure.

        Arguments:
        node -- the new node to insert
        """
        layer = node.get_max_layer()
        if self._nodes.get(layer) is None:
            self._nodes[layer] = list()
        
        self._nodes[layer].append(node)

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
        max_layer =  self._enter_point.get_max_layer()
        for layer in range(max_layer, layer - 1, -1): # Descend to the given layer
            logger.debug(f"Visiting layer {layer}, ep: {enter_point}")
            current_nearest_elements = self._search_layer_knn(query_node, [enter_point], 1, layer)
            logger.debug(f"Current nearest elements: {current_nearest_elements}")
            if len(current_nearest_elements) > 0:
                if enter_point.get_id() != query_node.get_id():
                    # get the nearest element to query node if the enter_point is not the query node itself
                    enter_point = self._find_nearest_element(query_node, current_nearest_elements)
            else: 
                logger.debug("First node in layer {}".format(layer))

        return enter_point

    def _assert_same_distance_algorithm(self, node):
        """Checks if the distance algorithm associated to node matches with the distance algorithm
        associated to the HNSW structure and raises HNSWUnmatchDistanceAlgorithmError when they do not match
        
        Arguments:
        node    -- the node to check
        """
        if node.get_distance_algorithm() != self.get_distance_algorithm():
            raise HNSWUnmatchDistanceAlgorithmError

    def _sanity_checks(self, node, check_empty: bool=True):
        """Raises HNSWUnmatchDistanceAlgorithmError or HNSWIsEmptyError exceptions, if necessary.

        Arguments:
        node        -- node to check
        check_empty -- flag to check if the Apotheosis structure is empty
        """
        # check if the distance algorithm is the same as the one associated to the node to delete
        self._assert_same_distance_algorithm(node)
        # check if it is empty
        if check_empty:
            self._assert_no_empty()
        return

    def insert(self, new_node):
        """Inserts a new node to the HNSW structure. On success, it return True
        Raises HNSWUnmatchDistanceAlgorithmError if the distance algorithm of the new node is distinct than 
        the distance algorithm associated to the HNSW structure.
        **WARNING** It does not check if the new node to insert already exists in the HNSW structure

        Arguments:
        new_node    -- the node to be added
        """

        self._sanity_checks(new_node, check_empty=False)

        enter_point = self._enter_point
        # Calculate the layer to which the new node belongs
        new_node_layer = int(-np.log(random.uniform(0,1)) * self._mL) // 1 # l in MY-TPAMI-20
        new_node.set_max_layer(new_node_layer)
        logger.info(f"New node in HNSW: \"{new_node.get_id()}\" (assigned level={new_node_layer})")
        
        if enter_point is not None:
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
        self._assert_layer_exists(layer) 
        self._assert_no_empty() 
       
        candidates_set = set(self._nodes[layer])
        if exclude_nodes is not None:
            logger.debug(f"Excluding nodes from random selection at L{layer}: <{[node.get_id() for node in exclude_nodes]}>")
            candidates_set = candidates_set - set(exclude_nodes)
        if len(candidates_set) == 0:
            logger.debug(f"No possible candidates for random choice at L{layer}, skipping ...")
            return None

        elm = random.choice(list(candidates_set))
        logger.debug(f"Random node chosen at L{layer}: \"{elm.get_id()}\"")
        return elm

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

    def _drunken_journey(self, layer: int, exclude_nodes: list) -> (None, bool):
        """Returns a random node at a given layer, excluding a certain set of nodes from selection set,
        and a boolean flag indicating whether or not a random walk was performed.
        
        Arguments:
        layer           -- layer level
        exclude_nodes   -- nodes to exclude
        """
        rand = random.random() # will be something in [0, 1)
        node = None
        flag = (rand <= self._beer_factor) # beer factor is the upper bound -> 1/2^(beer_factor to occur)
                                           # extremes are the same, so we only need to test half an interval
        if flag:
            logger.debug(f"Drunken journey at L{layer} ({rand})! Beer time ^^!")
            try:
                node = self._get_random_node_at_layer(layer, exclude_nodes=exclude_nodes)
            except HNSWLayerDoesNotExistError: # empty layer at the moment
                pass 
            finally:    
                if node is None: # not enough nodes in layer, adjust return flag to avoid problems
                    flag = False

        return node, flag

    def _shrink_nodes(self, nodes, layer):
        """Shrinks the maximum number of neighbors of nodes in a given layer.
        The maximum value depends on the layer (MMax0 for layer 0 or Mmax for other layers).

        Arguments:
        nodes   -- list of nodes to shrink
        layer   -- current layer to search neighbors and update in each node 
        """
        
        mmax = self._Mmax0 if layer == 0 else self._Mmax
        for node in nodes:
            _list = node.get_neighbors_at_layer(layer)
            if (len(_list) > mmax):
                shrinked_neighbors = self._select_neighbors_simple(node, _list, mmax)
                # this select must be simple, otherwise heuristics + extend_candidates can hold
                # and thus this list of neighs, used after to remove elements, can be incoherent
                # (if you want that, you must try/except the remove_neighbors below)
                # (we don't want that, right?)

                deleted_neighbors = list(set(_list) - set(shrinked_neighbors))
                for neigh in deleted_neighbors:
                    neigh.remove_neighbor(layer, node)
                    node.remove_neighbor(layer, neigh)
                
                node.set_neighbors_at_layer(layer, set(shrinked_neighbors))
                logger.debug(f"Node {node.get_id()} exceeded Mmax. New neighbors: {[n.get_id() for n in node.get_neighbors_at_layer(layer)]}")

    def _insert_node_to_layers(self, new_node, enter_point):
        """Inserts the new node from the minimum layer between HNSW enter point and the new node until layer 0.
        The first visited layer uses enter point as initial point for the best place to insert.

        Arguments:
        new_node    -- the node to insert
        enter_point -- the enter point to the first layer to visit 
        """
        
        min_layer = min(self._enter_point.get_max_layer(), new_node.get_max_layer())
        for layer in range(min_layer, -1, -1):
            currently_found_nn = self._search_layer_knn(new_node, enter_point, self._ef, layer)
            new_neighbors = self._select_neighbors(new_node, currently_found_nn, self._M, layer)
            # random walk (drunken journey)
            dj_visited_node, flag = self._drunken_journey(layer, exclude_nodes=new_neighbors)
            logger.debug(f"Found nn at L{layer}: {currently_found_nn}")
            if flag: # add random node visited
                new_neighbors.append(dj_visited_node)

            # connect both nodes bidirectionally
            for neighbor in new_neighbors: 
                neighbor.add_neighbor(layer, new_node)
                new_node.add_neighbor(layer, neighbor)
                logger.debug(f"Connections added at L{layer} between {new_node} and {neighbor}")

            # shrink (when we have exceeded the Mmax limit)
            self._shrink_nodes(new_neighbors, layer)
            #enter_point.extend(currently_found_nn)
            enter_point = currently_found_nn
   
    def _delete_neighbors_connections(self, node):
        """Given a node, deletes the connections to their neighbors.

        Arguments:
        node    -- the node to delete
        """

        logger.debug(f"Deleting neighbors of \"{node.get_id()}\"")
        for layer in range(node.get_max_layer() + 1):
            neighbors_to_remove = set()
            for neighbor in node.get_neighbors_at_layer(layer):
                logger.debug(f"Deleting at L{layer} link \"{neighbor.get_id()}\"")
                neighbors_to_remove.add(neighbor)
                
            for neighbor in neighbors_to_remove: # bidirectionally remove links
                node.remove_neighbor(layer, neighbor)
                neighbor.remove_neighbor(layer, node)

    def _delete_node_dict(self, node):
        """Deletes a node from the dict of the HNSW structure.
        It raises HNSWUndefinedError if the node to delete was not stored in the dict

        Arguments:
        node    -- the node to delete
        """
        layer = node.get_max_layer()
        try:
            self._nodes[layer].remove(node)
            if len(self._nodes[layer]) == 0: # remove this key, it is empty
                self._nodes.pop(layer)
        except:
            raise HNSWUndefinedError

    def delete(self, node):
        """Deletes a node from the dict of the HNSW structure.
        It raises HNSWUndefinedError if the node to delete was not stored in the dict
        Assumes the node exists in the structure

        Arguments:
        node    -- the node to delete
        """

        logger.info(f"Deleting \"node.get_id()\" in the HNSW structure ... ")
        if node.get_id() == self._enter_point.get_id(): # cover the case we try to delete enter point
            logger.debug("The node to delete is the enter point at the HNSW! Searching for a new enter point ...")
            self._delete_current_enter_point()

        # delete neighbor connections and the node itself from the node dict
        self._delete_neighbors_connections(node)
        self._delete_node_dict(node)
        return

    def _delete_current_enter_point(self):
        """Deletes the current enter point and establishes a new enter point
        (the first neighbor found at the highest possible layer).
        """
        
        logger.info("Deleting current HNSW enter point and updating it to a new enter point (if possible) ...")
        # iterate layers until we find the first neighbor at any layer
        current_ep = self._enter_point
        for layer in range(current_ep.get_max_layer(), -1, -1):
            neighs_list = current_ep.get_neighbors_at_layer(layer)
            if len(neighs_list) == 0:
                if layer == 0: # neighbors list is empty and we are at layer 0... check dict nodes, special case
                    if self._nodes.get(layer) is None or len(self._nodes[layer]) == 1: # the structure will be now empty
                        # it may happen the node is not at layer 0, as we only keep them in the max layer
                        logger.debug("No enter point! HNSW is now empty")
                        self._enter_point = None
                    else:
                        # it may happen we have other nodes in this layer, but not connected to found_node
                        # if so, select one of them randomly
                        self._enter_point = self._get_random_node_at_layer(layer, exclude_nodes=found_node)
                        logger.debug(f"New enter point randomly selected: \"{self._enter_point.get_id()}\"")

                continue # this layer is empty, continue until we find one layer with neighbors

            # force simple knn search
            closest_neighbor = self._select_neighbors_simple(current_ep, neighs_list, M=1)
            if len(closest_neighbor) == 1: # select this as new enter point and exit the loop
                self._enter_point = closest_neighbor.pop()
                break

        logger.debug("HNSW enter point updated!")

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
            neighbor_list = closest_node.get_neighbors_at_layer(layer)
            logger.debug(f"Neighbor list of closest node: {neighbor_list}")

            for neighbor in neighbor_list:
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
        working_candidates = candidates.copy() # makes a copy, otherwise we can get a modification in runtime
        if extend_candidates: # Neighbors of neighbors may be also my neighbors
            logger.debug(f"Initial candidate set: {candidates}")
            logger.debug("Extending candidates ...")
            for candidate in candidates:
                neighborhood_e = candidate.get_neighbors_at_layer(layer)
                for neighbor in neighborhood_e:
                    if neighbor.get_id() != node.get_id(): 
                        working_candidates.add(neighbor)

        logger.debug(f"Candidates list: {candidates}")
        
        discarded = set()
        while len(working_candidates) > 0 and len(_r) < M:
            # get nearest from W and from R and compare which is closer to new_node
            elm_nearest_W  = self._find_nearest_element(node, working_candidates)
            working_candidates.remove(elm_nearest_W)
            if len(_r) == 0: # trick for first iteration
                _r.add(elm_nearest_W)
                logger.debug(f"Adding {elm_nearest_W} to R")
                continue

            elm_nearest_R  = self._find_nearest_element(node, _r)
            logger.debug(f"Nearest_R vs nearest_W: {elm_nearest_R} vs {elm_nearest_W}")
            n2_is_closer_n1, _, _ = node.n2_closer_than_n1(n1=elm_nearest_R, n2=elm_nearest_W)
            if n2_is_closer_n1:
                _r.add(elm_nearest_W)
                logger.debug(f"Adding {elm_nearest_W} to R")
            else:
                discarded.add(elm_nearest_W)
                logger.debug(f"Adding {elm_nearest_W} to discarded set")

        if keep_pruned_conns:
            logger.debug("Keeping pruned connections ...")
            while len(discarded) > 0 and len(_r) < M:
                elm = self._find_nearest_element(node, discarded)
                discarded.remove(elm)
                
                _r.add(elm)
                logger.debug(f"Adding {elm} to R")

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
        if not self._distance_algorithm.is_spatial(): # similarity metric
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

    @classmethod
    def load_cfg_from_bytes(cls, byte_data: bytearray):
        """Loads a HNSW cfg from a byte data array.

        Arguments:
        byte_data  -- byte array containing HNSW configuration
        """

        logger.info("Reading HNSW from file ...")
        """https://docs.python.org/3/library/struct.html#struct-format-strings
        = means we want standard sizes (defined in common/constants.py)
        =I: unsigned int, 4B
        =d: double, 8B
        =c: char, 1B
        =?: bool, 1B
        """
        
        M       = struct.unpack('=I', byte_data[0:I_SIZE])[0]
        Mmax    = struct.unpack('=I', byte_data[I_SIZE:I_SIZE*2])[0]
        Mmax0   = struct.unpack('=I', byte_data[I_SIZE*2:I_SIZE*3])[0]
        ef      = struct.unpack('=I', byte_data[I_SIZE*3:I_SIZE*4])[0]
        mL      = struct.unpack('=d', byte_data[I_SIZE*4:(I_SIZE*4 + D_SIZE)])[0]
        
        current_ptr = (I_SIZE*4 + D_SIZE)
        distance_algorithm  = byte_data[current_ptr]
        # check distance_algorithm value and get the appropriate class for distance_algorithm field
        if distance_algorithm == TLSH:
            distance_algorithm = TLSHHashAlgorithm
        elif distance_algorithm == SSDEEP:
            distance_algorithm = SSDEEPHashAlgorithm
        else:
            raise ApotFileFormatUnsupportedError
        
        # XXX if C_SIZE != 1, this needs to be updated
        heuristic           = byte_data[current_ptr + C_SIZE] == 1 
        extend_candidates   = byte_data[current_ptr + C_SIZE*2] == 1
        keep_pruned_conns   = byte_data[current_ptr + C_SIZE*3] == 1
        current_ptr         += C_SIZE*3 + 1
        beer_factor         = struct.unpack('=d', byte_data[current_ptr:current_ptr + D_SIZE])[0]
        
        logger.debug("All parameters have been read. Creating now an empty HNSW ...") 
        new_HNSW = HNSW(M=M, Mmax=Mmax, Mmax0=Mmax0, ef=ef, distance_algorithm=distance_algorithm,\
                        heuristic=heuristic, extend_candidates=extend_candidates, keep_pruned_conns=keep_pruned_conns,\
                        beer_factor=beer_factor)
        # set other params programatically
        new_HNSW._mL = mL
        new_HNSW._set_queue_factor()
        logger.debug(f"HNSW configuration has been read and set: <{new_HNSW}>")
        
        return new_HNSW

    def serialize_cfg(self) -> bytearray:
        """Serializes the configuration of this HNSW.
        """
        bstr = bytearray()
        logger.info("Serializing HNSW configuration ...")
        bstr += struct.pack("=I", self._M)
        bstr += struct.pack("=I", self._Mmax)
        bstr += struct.pack("=I", self._Mmax0)
        bstr += struct.pack("=I", self._ef)
        bstr += struct.pack("=d", self._mL)
        
        # dump the distance algorithm associated to this structure
        # list of supported hashes is here
        if self._distance_algorithm == TLSHHashAlgorithm:
            bstr += struct.pack("=?", TLSH)
        elif self._distance_algorithm == SSDEEPHashAlgorithm:
            bstr += struct.pack("=?", SSDEEP)
        else:
            raise ApotFileFormatUnsupportedError
        
        bstr += struct.pack("=?", self._heuristic)
        bstr += struct.pack("=?", self._extend_candidates)
        bstr += struct.pack("=?", self._keep_pruned_conns)
        bstr += struct.pack("=d", self._beer_factor)
    
        logger.debug(f"HNSW configuration serialized correctly: {bstr}.")
        # save now the rest of stuff
        return bstr

    def dump(self, file, compress: bool=True):
        """Saves HNSW structure to permanent storage.
        pickle.dump may break with large data with SIGSEGV.
        
        Arguments:
        file    -- filename to save 
        """
        
        logger.info(f"Dumping to {file} (compressed? {compress}) ...")
        if compress:
            fp = io.BytesIO()
        else:
            fp = open(file, "wb")

        logger.debug(f"Pickling HNSW and dumping it ...")
        pickle.dump(self, fp, protocol=pickle.DEFAULT_PROTOCOL)
        
        # compress output
        if compress:
            compressed_data = gz.compress(fp.getvalue())
            with open(file, "wb") as fp:
                fp.write(compressed_data)
            logger.debug(f"Compressing memory file and saving it to {file} ... done!")
            fp.close()

    @classmethod
    def load(cls, file):
        """Restores HNSW structure from permanent storage.
        
        Arguments:
        file    -- filename to load
        """
        
        logger.info(f"Checking if {file} is compressed ...")
        # check if the file is compressed
        magic = b'\x1f\x8b\x08' # magic bytes of gzip file
        compressed = False
        with open(file, 'rb') as f:
            start_of_file = f.read(1024)
            f.seek(0)
            compressed = start_of_file.startswith(magic)

        # if compressed, load the appropriate file
        if not compressed:
            logger.debug(f"Not compressed. Desearializing it directly ...")
            with open(file, "rb") as f:
                obj = pickle.load(f)
        else:
            logger.debug(f"Compressed. Decompressing and desearializing ...")
            obj = pickle.load(gz.GzipFile(file))

        # check everything works as expected
        if not isinstance(obj, cls):
            raise TypeError(f"Expected an instance of {cls.__name__}, but got {type(obj).__name__}")
        return obj

    #TODO
    def get_thresholdnn_at_node(self, query, threshold):
        raise NotImplementedError

    def get_knn_node_at_layer(self, query, k, layer=0):
        """Returns the K-nearest neighbors of query node (at layer) as a dict, being the key the distance score

        Arguments:
        query   -- base node
        k       -- number of nearest neighbor to return
        layer   -- layer level
        """
        
        current_nearest_elements = query.get_neighbors_at_layer(0) 
        _knn_list = self._select_neighbors(query, current_nearest_elements, k, 0)
        _knn_dict = self._get_knndict_at_node(query, _knn_list)
        return _knn_dict

    def _get_knndict_at_node(self, query, node_list: list) -> dict:
        """Returns a dictionary of nodes where the key is the similarity score with the query node and the values are
        the corresponding nodes.

        Arguments:
        query       -- the base node
        node_list   -- the list of nodes to transform in dict
        """
        result = {}
        for node in node_list:
            value = node.calculate_similarity(query)
            if result.get(value) is None:
                result[value] = []
            result[value].append(node)
        return result

    def _expand_with_neighbors(self, nodes, layer=0):
        """Expands the set of nodes with their neighbors
        
        Arguments:
        nodes  -- set of nodes to expand
        layer  -- layer level (default 0)
        """
        result = set()
        for node in nodes:
            result.add(node)
            for neighbor in node.get_neighbors_at_layer(layer):
                result.add(neighbor)
        return result

    def aknn_search(self, query, k, ef=0): 
        """Performs an approximate k-nearest neighbors search using the HNSW structure.
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
        
        self._sanity_checks(query)

        # update ef to efConstruction, if necessary
        if ef == 0: 
            ef = self._ef

        logger.info(f"Performing an AKNN search of \"{query.get_id()}\" with ef={ef} ...")
        enter_point = self._descend_to_layer(query, layer=1) 
            
        # and now get the nearest elements
        current_nearest_elements = self._search_layer_knn(query, [enter_point], ef, 0)
        current_nearest_elements = self._expand_with_neighbors(current_nearest_elements)
        _knn_list = self._select_neighbors(query, current_nearest_elements, k, 0) 
        _knn_dict = self._get_knndict_at_node(query, _knn_list)
        logger.info(f"KNNs found (AKNN search): {_knn_dict} ...")
        
        # return a dictionary of nodes and similarity score
        return _knn_dict

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

        self._sanity_checks(query)
        
        ef = self._ef # exploration factor always set to default 

        logger.info(f"Performing a threshold search of \"{query.get_id()}\" with threshold {threshold} and nhops {n_hops} (ef={ef}) ...")
        # go to layer 0 (enter point is in layer 1)
        enter_point = self._descend_to_layer(query, layer=1)
        # get list of neighbors, considering threshold
        _current_neighs = self._search_layer_threshold(query, [enter_point], threshold, n_hops, layer=0)
        _knn_dict = self._get_knndict_at_node(query, _current_neighs)
        logger.info(f"KNNs found (threshold search): {_knn_dict} ...")

        # return a dictionary of nodes and similarity score
        return _knn_dict
    
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

    def _assign_node_color(self, node, colors, color_node_idx):
        color_node_map = ["blue", "yellow", "green", "red", "black"]

        if colors.get(node._module.id) is None:
            colors[node._module.id] = color_node_map[color_node_idx]
            color_node_idx = color_node_idx + 1 % len(color_node_map)
            logger.debug(f"Assigned color \"{colors[node._module.id]}\" to \"{node._module.internal_filename}\"")
        return colors[node._module.id], color_node_idx

    def draw(self, filename: str, show_distance: bool=True, format="pdf",\
                hash_subset: set=None, cluster: bool=False, threshold: float=0.0):
        """Creates a graph figure per level and saves it to a "L{level}filename" file.

        Arguments:
        filename        -- suffix filename to create (with extension)
        show_distance   -- to show the distance metric in the edges (default is True)
        format          -- matplotlib plt.savefig(..., format=format) (default is "pdf")
        hash_subset     -- hash subset to draw
        cluster         -- bool flag to draw also the structure in cluster mode (considering modules)
        threshold       -- float value to indicate the links between nodes to be drawn (only those with score >= threshold)
        """
        
        logger.info(f"Drawing HNSW with suffix \"{filename}\" ({format}; show distance? {show_distance}) -- hash subset: {hash_subset}")
        

        # iterate on layers
        for layer in sorted(self._nodes.keys(), reverse=True):
            colors = {}
            color_node_idx = 0
            node_colors = []
            features = {}

            G = nx.Graph()
            # iterate on nodes
            for node in self._nodes[layer]:
                node_label = node.get_id().replace(":", "")
                #if names.get(node_label) is None:
                node_features = node.get_draw_features()
                for key, value in node_features.items():
                    if key in features and isinstance(features[key], dict) and isinstance(value, dict):
                        features[key].update(value)
                    else:
                        features[key] = value

                # iterate on neighbors
                for neighbor in node.get_neighbors_at_layer(layer):
                    neigh_label = neighbor.get_id().replace(":", "")
                    #if names.get(neigh_label) is None:
                    node_features = neighbor.get_draw_features()
                    for key, value in node_features.items():
                        features[key].update(value)

                    edge_label = ""
                    # calculate similarity score to discriminate whether this link is drawn (depends on threshold value)
                    similarity_score = node.calculate_similarity(neighbor)
                    if show_distance:
                        edge_label = similarity_score

                    if self._distance_algorithm.is_spatial():
                        threshold_flag = similarity_score <= threshold or threshold == 0.0 # initial condition
                    else:
                        threshold_flag = similarity_score >= threshold
                    
                    # nodes are automatically created if they are not already in the graph
                    if hash_subset:
                        if node.get_id() in hash_subset and neighbor.get_id() in hash_subset and threshold_flag:
                            logger.debug(f"Both are in subset @L{layer}: {node.get_id()} -- {neighbor.get_id()}")
                            
                            G.add_edge(node_label, neigh_label, label=edge_label)
                    else:
                        # set color for the nodes, according to their associated modules 
                        if cluster:
                            color, color_node_idx = self._assign_node_color(node, colors, color_node_idx) 
                            if node_label not in G.nodes:
                                node_colors.append(color)
                            if neigh_label not in G.nodes:
                                color, color_node_idx = self._assign_node_color(neighbor, colors, color_node_idx) 
                                node_colors.append(color)
                        # add to graph
                        if threshold_flag:
                            G.add_edge(node_label, neigh_label, label=edge_label)
            if G.number_of_nodes() == 0: # this can happen when a subset is given
                logger.debug(f"L{layer} without nodes, skipping drawing ...")
                continue

            # set node attributes
            for key, value in features.items():
                nx.set_node_attributes(G, value, key)
            
            pos = nx.spring_layout(G, k=5)
            nx.draw(G, pos, node_size=1500, node_color='yellow', font_size=8, font_weight='bold', with_labels=True)
            nx.draw_networkx_edge_labels(G, pos, edge_labels = self._get_edge_labels(G), font_size=6)
            logger.debug(f"Saving graph to \"L{layer}{filename}\" file and generating DOT file ...")
            _str = f"L{layer}" + filename
            plt.savefig(_str, format=format)
            plt.clf()
            nx.drawing.nx_pydot.write_dot(G, _str + ".dot")
            
            if cluster:
                nx.set_node_attributes(G, nx.clustering(G), "cc")
                nx.draw(G, node_color=node_colors, node_size=[G.nodes[x]['cc']*1000 for x in G.nodes], with_labels=False)
                _str = f"L{layer}_clustering" + filename
                plt.savefig(_str, format=format)
                plt.clf()
                nx.drawing.nx_pydot.write_dot(G, _str + ".dot")
            

    # to support ==, now the object is not unhasheable (cannot be stored in sets or dicts)
    def __eq__(self, other):
        """Returns True if this object and other are the same, False otherwise.
        
        Arguments:
        other   -- HNSW to check
        """
        logger.info(f"Comparing {self} with {other} ...")
        if type(self) != type(other):
            return False

        logger.debug("Comparing attributes length ...")
        self_attbs = self.__dict__
        other_attbs = other.__dict__
        if len(self_attbs) != len(other_attbs):
            return False
        
        try:
            logger.debug("Comparing HNSW configuration ...")
            # HNSW configuration
            equal = self._M == other._M and\
                        self._Mmax == other._Mmax and\
                        self._Mmax0 == other._Mmax0 and\
                        self._ef == other._ef and\
                        self._mL == other._mL and\
                        self._distance_algorithm == other._distance_algorithm and\
                        self._queue_factor == other._queue_factor and\
                        self._heuristic == other._heuristic and\
                        self._extend_candidates == other._extend_candidates and\
                        self._keep_pruned_conns == other._keep_pruned_conns and\
                        self._beer_factor == other._beer_factor
            
            if not equal:
                return False
            
            logger.debug("Comparing enter points ...")
            # same enter point?
            same_ep = self._enter_point.is_equal(other._enter_point)
            
            logger.debug("Comparing nodes dict ...")
            # now, check the node dicts...
            if len(self._nodes) != len(other._nodes):
                return False
            for layer in self._nodes:
                logger.debug(f"Comparing nodes at L{layer} ...")
                # get pageids from layer
                self_pageids = set([node.get_internal_page_id() for node in self._nodes[layer]])
                other_pageids = set([node.get_internal_page_id() for node in other._nodes[layer]])
                if self_pageids != other_pageids:
                    logger.debug("Different sets found at L{layer}: {self_pageids} vs {other_pageids}")
                    return False
            
            return True
        except Exception as e:
            logger.debug("Exception occured in __eq__: {e}")
            return False
            

    def __str__(self):
        """Printing utility, prints the configuration of this HNSW object.
        """
        attbs_dict = self.__dict__
        _str = ""
        for k, v in attbs_dict.items():
            if k == "_enter_point" and v:
                _str += k + f": {str(v.get_id())}; "
            elif k == "_nodes" and v:
                _str += k + f": <list of nodes per layer> {len(v)} nodes, hidden; "
            elif k == "_distance_algorithm":
                _str += k + f": {v.__name__}; "
            else:
                _str += k + f": {str(v)}; "
        
        return _str[:-2] # remove last "; " from the string

# unit test
# run this as "python3 -m datalayer.hnsw"
import common.utilities as util
from datalayer.node.hash_node import HashNode
from datalayer.hash_algorithm.tlsh_algorithm import TLSHHashAlgorithm
from datalayer.hash_algorithm.ssdeep_algorithm import SSDEEPHashAlgorithm

if __name__ == "__main__":
    parser = util.configure_argparse()
    args = parser.parse_args()

    util.configure_logging(args.loglevel.upper()) 
    # Create an HNSW structure
    myHNSW = HNSW(M=args.M, ef=args.ef, Mmax=args.Mmax, Mmax0=args.Mmax0,\
                    beer_factor=args.beer_factor,
                    heuristic=args.heuristic, extend_candidates=not args.no_extend_candidates, keep_pruned_conns=not args.no_keep_pruned_conns,\
                    distance_algorithm=TLSHHashAlgorithm)

    # Create the nodes based on TLSH Fuzzy Hashes
    node1 = HashNode("T12B81E2134758C0E3CA097B381202C62AC793B46686CD9E2E8F9190EC89C537B5E7AF4C", TLSHHashAlgorithm)
    node2 = HashNode("T10381E956C26225F2DAD9D5C2C5C1A337FAF3708A25012B8A1EACDAC00B37D557E0E714", TLSHHashAlgorithm)
    node3 = HashNode("T1DF8174A9C2A506F9C6FFC292D6816333FEF1B845C419121A0F91CF5359B5B21FA3A304", TLSHHashAlgorithm)
    node4 = HashNode("T1DF8174A9C2A506F9C6FFC292D6816333FEF1B845C419121A0F91CF5359B5B21FA3A304", TLSHHashAlgorithm)
    node5 = HashNode("T1DF8174A9C2A506F9C6FFC292D6816333FEF1B845C419121A0F91CF5359B5B21FA3A305", TLSHHashAlgorithm)
    nodes = [node1, node2, node3]

    print("Testing insert ...")
    # Insert nodes on the HNSW structure
    if myHNSW.insert(node1):
        print(f"Node \"{node1.get_id()}\" inserted correctly.")
    if myHNSW.insert(node2):
        print(f"Node \"{node2.get_id()}\" inserted correctly.")
    if myHNSW.insert(node3):
        print(f"Node \"{node3.get_id()}\" inserted correctly.")
    if myHNSW.insert(node4):
        print(f"Node \"{node4.get_id()}\" inserted correctly (twice).")

    print(f"Enter point: {myHNSW.get_enter_point()}")

    # Perform k-nearest neighbor search based on TLSH fuzzy hash similarity
    query_node = HashNode("T1BF81A292E336D1F68224D4A4C751A2B3BB353CA9C2103BA69FA4C7908761B50F22E301", TLSHHashAlgorithm)
    for node in nodes:
        print(node, "Similarity score: ", node.calculate_similarity(query_node))

    print('Testing approximate aknn_search ...')
    try:
        results = myHNSW.aknn_search(query_node, k=2, ef=4)
        print("Total neighbors found: ", len(results))
        util.print_results(results)
    except HNSWIsEmptyError:
        print("ERROR: performing a KNN search in an empty HNSW")
        
    print('Testing threshold_search ...')
    # Perform threshold search to retrieve nodes above a similarity threshold
    try:
        results = myHNSW.threshold_search(query_node, threshold=220, n_hops=3)
        util.print_results(results, show_keys=True)
    except HNSWIsEmptyError:
        print("ERROR: performing a KNN search in an empty HNSW")

    # Draw it
    myHNSW.draw("unit_test.pdf")

    # Dump created HNSW structure to disk
    myHNSW.dump("myHNSW.txt")

    # Restore HNSW structure from disk
    myHNSW = HNSW.load("myHNSW.txt")

