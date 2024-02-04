import logging
logger = logging.getLogger(__name__)

from datalayer.trie_hash import TrieHash
from datalayer.hnsw import HNSW

# custom exceptions
from datalayer.errors import NodeNotFoundError
from datalayer.errors import NodeAlreadyExistsError

from datalayer.errors import ApotheosisUnmatchDistanceAlgorithmError
from datalayer.errors import ApotheosisIsEmptyError

__author__ = "Daniel Huici Meseguer and Ricardo J. Rodríguez"
__copyright__ = "Copyright 2024"
__credits__ = ["Daniel Huici Meseguer", "Ricardo J. Rodríguez"]
__license__ = "GPL"
__version__ = "0.3"
__maintainer__ = "Daniel Huici"
__email__ = "reverseame@unizar.es"
__status__ = "Development"

# file extensions
HNSW_FILEEXT = ".hnsw"
TRIE_FILEEXT = ".trie"

class Apotheosis:
    
    def __init__(self, M=0, ef=0, Mmax=0, Mmax0=0,
                    distance_algorithm=None,
                    heuristic=False, extend_candidates=True, keep_pruned_conns=True,
                    prefix_filename=None):
        """Default constructor."""
        if prefix_filename == None:
            # construct both data structures (a HNSW and a trie for all nodes -- will contain @HashNode)
            self._HNSW = HNSW(M, ef, Mmax, Mmax0, distance_algorithm, heuristic, extend_candidates, keep_pruned_conns)
            self._distance_algorithm = distance_algorithm
            # trie for all nodes (of @HashNode)
            self._trie = TrieHash(distance_algorithm)
        else: # load the structure from prefix_filename
            self._HNSW = HNSW.load(prefix_filename + HNSW_FILEEXT)
            self._distance_algorithm = self._HNSW.get_distance_algorithm()
            self._trie = TrieHash.load(prefix_filename + TRIE_FILEEXT)
            # check if both structures have been generated with the same distance algorithm 
            if type(self._trie.get_hash_algorithm()) != type(self._distance_algorithm):
                raise ApotheosisUnmatchDistanceAlgorithmError

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
        
    def add_node(self, new_node):
        """Adds a new node to the Apotheosis structure. On success, it return True
        Raises ApotheosisUnmatchDistanceAlgorithmError if the distance algorithm of the new node is distinct than 
        the distance algorithm associated to the HNSW structure.
        Raises NodeAlreadyExistsError if the there is a node with the same ID as the new node.
        
        Arguments:
        new_node    -- the node to be added
        """
        
        self._sanity_checks(new_node, check_empty=False)
   
        logger.info(f"Adding node \"{new_node.get_id()}\"  ...")        
        # adding the node to the trie may raise exception NodeAlreadyExistsError 
        self._trie.insert(new_node)     # O(self._distance_algorithm.get_max_hash_alphalen())
        self._HNSW.add_node(new_node)   # N*(log N), see Section 4.2.2 in MY-TPAMI-20
        logger.info(f"Node \"{new_node.get_id()}\" correctly added!")        
        return True

    def delete_node(self, node):
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

        logger.info(f"Deleting node \"{node.get_id()} ...\"")        
        # search the node in the trie structure
        is_found, found_node = self._trie.search(node.get_id()) 
        if is_found:
            logger.debug(f"Node \"{node.get_id()}\" found! Deleting it ...")
            self._HNSW.delete_node(found_node)
        else:
            logger.debug(f"Node \"{node.get_id()}\" not found!")
            raise NodeNotFoundError

        return True

    def dump(self, prefix_filename):
        """Saves Apotheosis structure to permanent storage.

        Arguments:
        prefix_filename -- prefix filename to save 
        """

        logger.info(f"Saving Apotheosis structure to disk (prefix filename \"{prefix_filename}\") ...")
        self._HNSW.dump(prefix_filename + HNSW_FILEEXT)
        self._trie.dump(prefix_filename + TRIE_FILEEXT)
        return

    @classmethod
    def load(cls, prefix_filename):
        """Restores Apotheosis structure from permanent storage.
        
        Arguments:
        prefix_filename -- prefix filename to load
        """
        
        logger.info(f"Restoring Apotheosis structure from disk (prefix filename \"{prefix_filename}\") ...")
        newAPO = Apotheosis(prefix_filename=prefix_filename)
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
        _exact, _node = self._trie.search(query.get_id())       # O(self._distance_algorithm.get_max_hash_alphalen())
        if _exact: # get k-nn at layer 0, using HNSW structure
            # as node exists, this call is safe
            logger.debug(f"Node \"{query.get_id()}\" found in the trie! Recovering now its neighbors from HNSW ... ")
            _knn_dict = self._HNSW.get_knn_at_node(_node, k) 
        else: # get approximate k-nns with HNSW search
            logger.debug(f"Node \"{query.get_id()}\" NOT found in the trie! Recovering now its approximate neighbors ... ")
            _knn_dict = self._HNSW.aknn_search(query, k, ef)    # log N, see Section 4.2.1 in MY-TPAMI-20

        return _exact, _knn_dict

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
       
        self._sanity_checks(node)
        
        logger.info(f"Performing a threshold search for \"{query.get_id()}\" (threshold={threshold}, n_hops={n_hops})")
        _exact, _node = self._trie.search(query.get_id())
        if _exact: # get k-nn at layer 0, using HNSW structure
            # as node exists, this is safe
            logger.debug(f"Node \"{query.get_id()}\" found in the trie! Recovering now its neighbors ... ")
            _knn_dict = self._HNSW.get_thresholdnn_at_node(query, k) 
        else: # get approximate k-nns with HNSW search
            logger.debug(f"Node \"{query.get_id()}\" NOT found in the trie! Recovering now its approximate neighbors ... ")
            _knn_dict = self._HNSW.threshold_search(query, threshold, n_hops)

        return _exact, _knn_dict

    def draw(self, filename: str, show_distance: bool=True, format="pdf"):
        """Creates a graph figure per level of the HNSW structure and saves it to a filename file.

        Arguments:
        filename        -- filename to create (with extension)
        show_distance   -- to show the distance metric in the edges (default is True)
        format          -- file extension
        """
        self._HNSW.draw(filename, show_distance, format)

# unit test
import common.utilities as util
from datalayer.node.node_hash import HashNode
from datalayer.hash_algorithm.tlsh_algorithm import TLSHHashAlgorithm

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
    node1 = HashNode("T12B81E2134758C0E3CA097B381202C62AC793B46686CD9E2E8F9190EC89C537B5E7AF4C", TLSHHashAlgorithm)
    node2 = HashNode("T10381E956C26225F2DAD9D5C2C5C1A337FAF3708A25012B8A1EACDAC00B37D557E0E714", TLSHHashAlgorithm)
    node3 = HashNode("T1DF8174A9C2A506F9C6FFC292D6816333FEF1B845C419121A0F91CF5359B5B21FA3A304", TLSHHashAlgorithm)
    node4 = HashNode("T1DF8174A9C2A506F9C6FFC292D6816333FEF1B845C419121A0F91CF5359B5B21FA3A304", TLSHHashAlgorithm)
    node5 = HashNode("T1DF8174A9C2A506F9C6FFC292D6816333FEF1B845C419121A0F91CF5359B5B21FA3A305", TLSHHashAlgorithm)
    nodes = [node1, node2, node3]

    print("Testing add_node ...")
    # Insert nodes on the HNSW structure
    if myAPO.add_node(node1):
        print(f"Node \"{node1.get_id()}\" inserted correctly.")
    if myAPO.add_node(node2):
        print(f"Node \"{node2.get_id()}\" inserted correctly.")
    if myAPO.add_node(node3):
        print(f"Node \"{node3.get_id()}\" inserted correctly.")
    try:
        myAPO.add_node(node4)
        print(f"WRONG --> Node \"{node4.get_id()}\" inserted correctly.")
    except NodeAlreadyExistsError:
        print(f"Node \"{node4.get_id()}\" cannot be inserted, already exists!")

    print(f"Enter point: {myAPO.get_HNSW_enter_point()}")

    # draw it
    if args.draw:
        myAPO.draw("unit_test.pdf")

    try:
        myAPO.delete_node(node5)
    except NodeNotFoundError:
        print(f"Node \"{node5.get_id()}\" not found!")
   
    print("Testing delete_node ...")
    myAPO.delete_node(node1)
    #myHNSW.delete_node(node2)
    #myHNSW.delete_node(node3)

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

    # Dump created Apotheosis structure to disk
    myAPO.dump("myAPO")

    # Restore Apotheosis structure from disk
    myAPO = Apotheosis.load("myAPO")

