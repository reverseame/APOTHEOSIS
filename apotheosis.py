import logging
logger = logging.getLogger(__name__)

__author__ = "Daniel Huici Meseguer and Ricardo J. Rodríguez"
__copyright__ = "Copyright 2024"
__credits__ = ["Daniel Huici Meseguer", "Ricardo J. Rodríguez"]
__license__ = "GPL"
__version__ = "0.3"
__maintainer__ = "Daniel Huici"
__email__ = "reverseame@unizar.es"
__status__ = "Development"

from datalayer.radix_hash import RadixHash
from datalayer.hnsw import HNSW

# custom exceptions
from datalayer.errors import NodeNotFoundError
from datalayer.errors import NodeAlreadyExistsError

from datalayer.errors import ApotheosisUnmatchDistanceAlgorithmError
from datalayer.errors import ApotheosisIsEmptyError

# preferred file extension
PREFERRED_FILEEXT = ".apo"

class Apotheosis:
    
    def __init__(self, M=0, ef=0, Mmax=0, Mmax0=0,
                    distance_algorithm=None,
                    heuristic=False, extend_candidates=True, keep_pruned_conns=True,
                    filename=None):
        """Default constructor."""
        if filename == None:
            # construct both data structures (a HNSW and a radix tree for all nodes -- will contain @HashNode)
            self._HNSW = HNSW(M, ef, Mmax, Mmax0, distance_algorithm, heuristic, extend_candidates, keep_pruned_conns)
            self._distance_algorithm = distance_algorithm
            # radix hash tree for all nodes (of @HashNode)
            self._radix = RadixHash(distance_algorithm)
        else:
            # load structures from filename
            self._HNSW = HNSW.load(filename)
            self._distance_algorithm = self._HNSW.get_distance_algorithm()
            self._radix = RadixHash(self._distance_algorithm, self._HNSW)

    def get_distance_algorithm(self):
        """Getter for _distance_algorithm"""
        return self._distance_algorithm

    def _assert_same_distance_algorithm(self, node):
        """Checks if the distance algorithm associated to node matches with the distance algorithm
        associated to the Apotheosis structure and raises ApotheosisUnmatchDistanceAlgorithmError when they do not match

        Arguments:
        node    -- the node to check
        """
        if node.get_distance_algorithm() != self.get_distance_algorithm():
             raise ApotheosisUnmatchDistanceAlgorithmError
    
    def _assert_no_empty(self):
        """Raises ApotheosisIsEmptyError if the Apotheosis structure is empty."""
        if self._HNSW._is_empty():
            raise ApotheosisIsEmptyError

    def get_HNSW_enter_point(self):
        """Returns the enter point of the HNSW structure.
        """
        return self._HNSW.get_enter_point()
        
    def insert(self, new_node):
        """Inserts a new node to the Apotheosis structure. On success, it return True
        Raises ApotheosisUnmatchDistanceAlgorithmError if the distance algorithm of the new node is distinct than 
        the distance algorithm associated to the HNSW structure.
        Raises NodeAlreadyExistsError if the there is a node with the same ID as the new node.
        
        Arguments:
        new_node    -- the node to be inserted
        """
        
        self._sanity_checks(new_node, check_empty=False)
   
        logger.info(f"Inserting node \"{new_node.get_id()}\"  ...")        
        # adding the node to the radix tree may raise exception NodeAlreadyExistsError 
        self._radix.insert(new_node)    # O(len(new_node.get_id()))
        self._HNSW.insert(new_node)     # N*(log N), see Section 4.2.2 in MY-TPAMI-20
        logger.info(f"Node \"{new_node.get_id()}\" correctly added!")        
        return True

    def delete(self, node):
        """Deletes a node of the Apotheosis structure. On success, it returns True
        It may raise several exceptions:
            * ApotheosisIsEmptyError when the HNSW structure has no nodes.
            * ApotheosisUnmatchDistanceAlgorithmError when the distance algorithm of the node to delete
              does not match the distance algorithm associated to the HNSW structure.
            * NodeNotFoundError when the node to delete is not found in the Apotheosis structure.
            * HNSWUndefinedError when no neighbor is found at layer 0 (shall never happen this!).
        
        Arguments:
        node    -- the node to delete
        """
        self._sanity_checks(node)

        logger.info(f"Deleting node \"{node.get_id()}\" Trying first removing it in the radix tree ...")        
        found_node = self._radix.delete(node.get_id())
        if found_node is not None:
            logger.debug(f"Node \"{node.get_id()}\" found in the radix tree! Deleting it now in the HNSW ...")
            self._HNSW.delete(found_node)
        else:
            logger.debug(f"Node \"{node.get_id()}\" not found in the radix tree!")
            raise NodeNotFoundError

        return True

    def dump(self, filename: str, compress: bool=True):
        """Saves Apotheosis structure to permanent storage.

        Arguments:
        filename    -- filename to save
        TODO
        """

        logger.info(f"Saving Apotheosis structure to disk (filename \"{filename}\") ...")
        self._HNSW.dump(filename, compress)
        return

    @classmethod
    def load(cls, filename):
        """Restores Apotheosis structure from permanent storage.
        
        Arguments:
        filename    -- filename to load
        """
        
        logger.info(f"Restoring Apotheosis structure from disk (filename \"{filename}\") ...")
        newAPO = Apotheosis(filename=filename)
        return newAPO

    def _sanity_checks(self, node, check_empty: bool=True):
        """Raises ApotheosisUnmatchDistanceAlgorithmError or ApotheosisIsEmptyError exceptions, if necessary.

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

    def knn_search(self, query, k, ef=0):
        """If query is present in the Apotheosis structure, returns True and the K nearest neighbors to query. 
        Otherwise, returns False and the approximate K nearest neighbors to query.
        It raises the following exceptions:
            * ApotheosisUnmatchDistanceAlgorithmError if the distance algorithm of the new node is distinct than 
              the distance algorithm associated to the HNSW structure.
            * ApotheosisIsEmptyError if the HNSW structure is empty

        Arguments:
        query   -- base node
        k       -- number of nearest neighbors to query node to return
        ef      -- exploration factor (search recall)
        """
        
        self._sanity_checks(query)
        
        logger.info(f"Performing a KNN search for \"{query.get_id()}\" (k={k}, ef={ef})")
        exact, node = self._radix.search(query.get_id())      # O(len(query.get_id()))
        if exact: # get k-nn at layer 0, using HNSW structure
            # as node exists, this call is safe
            logger.debug(f"Node \"{query.get_id()}\" found in the radix tree! Recovering now its neighbors from HNSW ... ")
            knn_dict = self._HNSW.get_knn_node_at_layer(node, k, layer=0) 
        else: # get approximate k-nns with HNSW search
            logger.debug(f"Node \"{query.get_id()}\" NOT found in the radix tree! Recovering now its approximate neighbors ... ")
            knn_dict = self._HNSW.aknn_search(query, k, ef)    # log N, see Section 4.2.1 in MY-TPAMI-20

        return exact, knn_dict

    def threshold_search(self, query, threshold, n_hops):
        """Performs a threshold search to retrieve nodes that satisfy a certain similarity threshold using the HNSW structure.
        It returns a list of nearest neighbor nodes to query that satisfy the specified similarity threshold.
        It raises the following exceptions:
            * ApotheosisUnmatchDistanceAlgorithmError if the distance algorithm of the new node is distinct than 
              the distance algorithm associated to the HNSW structure.
            * ApotheosisIsEmptyError if the HNSW structure is empty

        Arguments:
        query      -- the query node for which to find the neighbors with a similarity above the given percentage
        threshold  -- the similarity threshold to satisfy 
        n_hops     -- number of hops to perform from each nearest neighbor
        """
       
        self._sanity_checks(query)
        
        logger.info(f"Performing a threshold search for \"{query.get_id()}\" (threshold={threshold}, n_hops={n_hops})")
        exact, node = self._radix.search(query.get_id())
        if exact: # get k-nn at layer 0, using HNSW structure
            # as node exists, this is safe
            logger.debug(f"Node \"{query.get_id()}\" found in the radix tree! Recovering now its neighbors ... ")
            knn_dict = self._HNSW.get_thresholdnn_at_node(query, threshold) 
        else: # get approximate k-nns with HNSW search
            logger.debug(f"Node \"{query.get_id()}\" NOT found in the radix tree! Recovering now its approximate neighbors ... ")
            knn_dict = self._HNSW.threshold_search(query, threshold, n_hops)

        return exact, knn_dict

    def draw_hashes_subset(self, hash_set: set(), filename: str, show_distance: bool=True, format="pdf"):
        """Creates a graph figure per level of the HNSW structure and saves it to a filename file, 
        but only considering hash values in hash_set.

        Arguments:
        hash_set        -- set of nodes to draw
        filename        -- filename to create (with extension)
        show_distance   -- to show the distance metric in the edges (default is True)
        format          -- matplotlib plt.savefig(..., format=format) (default is "pdf")
        """
        
        logger.info(f"Drawing to {filename} (subset: {hash_set}) ...")
        self._HNSW.draw(filename, show_distance=show_distance, format=format, hash_subset=hash_set)

    def draw(self, filename: str, show_distance: bool=True, format="pdf", cluster: bool=False):
        """Creates a graph figure per level of the HNSW structure and saves it to a filename file.

        Arguments:
        filename        -- filename to create (with extension)
        show_distance   -- to show the distance metric in the edges (default is True)
        format          -- matplotlib plt.savefig(..., format=format) (default is "pdf")
        cluster         -- bool flag to draw also the structure in cluster mode (considering modules)
        """
        self._HNSW.draw(filename, show_distance=show_distance, format=format, cluster=cluster)

# unit test
import common.utilities as util
from datalayer.node.hash_node import HashNode
from datalayer.hash_algorithm.tlsh_algorithm import TLSHHashAlgorithm
from random import random
import math

def rand(apo: Apotheosis):
    upper_limit = myAPO.get_distance_algorithm().get_max_hash_alphalen()
    return _rand(upper_limit)

def _rand(upper_limit: int=1):
    lower_limit = 0
    return math.floor(random()*(upper_limit - lower_limit) + lower_limit)


def search_knns(apo, query_node):
    try:
        exact_found, results = apo.knn_search(query_node, k=2, ef=4)
        print(f"{query_node.get_id()} exact found? {exact_found}")
        print("Total neighbors found: ", len(results))
        util.print_results(results)
    except ApotheosisIsEmptyError:
        print("ERROR: performing a KNN search in an empty Apotheosis structure")

if __name__ == "__main__":
    parser = util.configure_argparse()
    args = parser.parse_args()
    util.configure_logging(args.loglevel.upper())

    # Create an Apotheosis structure
    myAPO = Apotheosis(M=args.M, ef=args.ef, Mmax=args.Mmax, Mmax0=args.Mmax0,\
                    heuristic=args.heuristic, extend_candidates=not args.no_extend_candidates, keep_pruned_conns=not args.no_keep_pruned_conns,\
                    distance_algorithm=TLSHHashAlgorithm)

    # Create the nodes based on TLSH Fuzzy Hashes
    hash1 = "T1BF81A292E336D1F68224D4A4C751A2B3BB353CA9C2103BA69FA4C7908761B50F22E301" #fake
    hash2 = "T12B81E2134758C0E3CA097B381202C62AC793B46686CD9E2E8F9190EC89C537B5E7AF4C" 
    hash3 = "T10381E956C26225F2DAD9D5C2C5C1A337FAF3708A25012B8A1EACDAC00B37D557E0E714"
    hash4 = "T1DF8174A9C2A506F9C6FFC292D6816333FEF1B845C419121A0F91CF5359B5B21FA3A304" 
    hash5 = "T1DF8174A9C2A506F9C6FFC292D6816333FEF1B845C419121A0F91CF5359B5B21FA3A305" #fake
    hash6 = "T1DF8174A9C2A506FC122292D644816333FEF1B845C419121A0F91CF5359B5B21FA3A305" #fake
    hash7 = "T10381E956C26225F2DAD9D097B381202C62AC793B37082B8A1EACDAC00B37D557E0E714" #fake

    node1 = HashNode(hash1, TLSHHashAlgorithm)
    node2 = HashNode(hash2, TLSHHashAlgorithm)
    node3 = HashNode(hash3, TLSHHashAlgorithm)
    node4 = HashNode(hash4, TLSHHashAlgorithm)
    node5 = HashNode(hash5, TLSHHashAlgorithm)
    nodes = [node1, node2, node3]

    print("Testing insert ...")
    # Insert nodes on the HNSW structure
    if myAPO.insert(node1):
        print(f"Node \"{node1.get_id()}\" inserted correctly.")
    if myAPO.insert(node2):
        print(f"Node \"{node2.get_id()}\" inserted correctly.")
    if myAPO.insert(node3):
        print(f"Node \"{node3.get_id()}\" inserted correctly.")
    try:
        myAPO.insert(node4)
        print(f"WRONG --> Node \"{node4.get_id()}\" inserted correctly.")
    except NodeAlreadyExistsError:
        print(f"Node \"{node4.get_id()}\" cannot be inserted, already exists!")

    print(f"Enter point: {myAPO.get_HNSW_enter_point()}")

    # draw it
    if args.draw:
        myAPO.draw("unit_test.pdf")

    try:
        myAPO.delete(node5)
    except NodeNotFoundError:
        print(f"Node \"{node5.get_id()}\" not found!")

    print("Testing delete ...")
    if myAPO.delete(node1):
        print(f"Node \"{node1.get_id()}\" deleted!")

    # Perform k-nearest neighbor search based on TLSH fuzzy hash similarity
    query_node = HashNode("T1BF81A292E336D1F68224D4A4C751A2B3BB353CA9C2103BA69FA4C7908761B50F22E301", TLSHHashAlgorithm)
    for node in nodes:
        print(node, "Similarity score: ", node.calculate_similarity(query_node))

    print('Testing knn_search ...')
   
    search_knns(myAPO, node1)
    search_knns(myAPO, node5)
    print('Testing threshold_search ...')
    # Perform threshold search to retrieve nodes above a similarity threshold
    try:
        exact_found, results = myAPO.threshold_search(query_node, threshold=220, n_hops=3)
        print(f"{query_node.get_id()} exact found? {exact_found}")
        util.print_results(results, show_keys=True)
    except ApotheosisIsEmptyError:
        print("ERROR: performing a KNN search in an empty Apotheosis structure")

    from datetime import datetime
    now = datetime.now()
    date_time = now.strftime("%H:%M:%S")
    # Dump created Apotheosis structure to disk
    print(f"Saving    Apotheosis at {date_time} ...")
    myAPO.dump("myAPO"+date_time)
    myAPO.dump("myAPO_uncompressed"+date_time, compress=False)

    # Restore Apotheosis structure from disk
    print(f"Restoring Apotheosis at {date_time} ...")
    myAPO = Apotheosis.load("myAPO_uncompressed"+date_time)
    myAPO = Apotheosis.load("myAPO"+date_time)

    # cluster test
    in_cluster = 10 # random nodes in the cluster
    alphabet = []
    for i in range(0, 10): # '0'..'9'
        alphabet.append(str(i + ord('0')))
    
    for i in range(0, 6): # 'A'..'F'
        alphabet.append(str(i + ord('0')))

    for i in range(0, in_cluster*100):
        limit = 0
        while limit <= 2:
            limit = _rand(len(alphabet))

        if random() >= .5: # 50%
            _hash = hash1
        else:
            _hash = hash2
        
        _hash = _hash[0:limit - 1] + alphabet[_rand(len(alphabet))] + _hash[limit + 1:]
        node = HashNode(_hash, TLSHHashAlgorithm)
        try:
            myAPO.insert(node)
        except:
            continue

    myAPO.draw_hashes_subset([node.get_id() for node in nodes], "unit_test_cluster.pdf")


