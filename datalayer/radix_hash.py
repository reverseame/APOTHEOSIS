import pickle
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

from datalayer.node.hash_node import HashNode
from datalayer.node.radix_node import RadixHashNode

# for loading from file
from datalayer.hnsw import HNSW
from common.errors import HNSWLayerDoesNotExistError

logging.getLogger('pickle').setLevel(logging.WARNING)

class RadixHash:

    def __init__(self, hash_algorithm, HNSW: HNSW=None):
        """Default constructor.

        Arguments:
        hash_algorithm  -- hash algorithm that creates the hashes stored in the radix tree
        """
        self._hash_algorithm = hash_algorithm
        self._root = RadixHashNode()
        if HNSW:
            # recreate the radix tree from the incoming HNSW
            max_layer = HNSW.get_enter_point().get_max_layer()
            logger.debug(f"Recreating from existing HNSW. Starting at L{max_layer}...")
            # iterate in layers, top-down
            for layer in range(max_layer, -1, -1):
                try:
                    nodes = HNSW.get_nodes_at_layer(layer)
                    # iterate on nodes
                    for node in nodes:
                        self.insert(node)
                except HNSWLayerDoesNotExistError:
                    logger.debug(f"L{layer} is empty. Skipping ...")
                    continue

    def get_hash_algorithm(self):
        """Getter for _hash_algorithm."""
        return self._hash_algorithm

    def insert(self, hash_node: HashNode):
        """Inserts a hash node into the tree

        Arguments:
        hash_node   -- hash node to insert
        """
       
        logging.info(f"Inserting \"{hash_node.get_id()}\" in the radix hash tree ... ")
        self._root.insert(hash_node.get_id(), hash_node)

    def search(self, hash_value: str) -> (bool, HashNode):
        """Returns True and the associated hash node if the hash value is on the radix hash tree, (False, None) otherwise

        Arguments:
        hash_value  -- hash value to check
        """
        found, hash_node = self._root.search(hash_value, None)
        logging.info(f"Searching \"{hash_value}\" in the radix hash tree ... Found? {found}")
        return found, hash_node

    def delete(self, hash_value: str) -> HashNode:
        """Deletes a hash value from the radix hash tree, if it exists.
            Returns the HashNode matching with the word if it was found and deleted, None otherwise
        
        Arguments:
        word    -- word to delete
        """

        found_node = self._root.delete(hash_value)
        logging.info(f"Deleting \"{hash_value}\" in the radix hash tree ... Found and deleted? {found_node is not None}")
        return found_node

    def print_tree(self, height: int = 0, only_hashes=False):
        """Prints the radix tree showing its internal structure and/or contained data.

        Arguments:
        height      -- height of the printed node
        only_hashes -- bool flag to indicate if we only need to print hashes contained in the tree
        """
        self._root.print_tree(height, only_hashes)
    
    def dump(self, file):
        """Saves radix hash tree structure to permanent storage.

        Arguments:
        file    -- filename to save
        """

        with open(file, "wb") as f:
            pickle.dump(self, f, protocol=pickle.DEFAULT_PROTOCOL)

    @classmethod
    def load(cls, file):
        """Restores radix hash tree structure from permanent storage.

        Arguments:
        file    -- filename to load
        """
        with open(file, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected an instance of {cls.__name__}, but got {type(obj).__name__}")
        return obj

# unit test
# run this as "python3 -m datalayer.radix_hash"
import random
import common.utilities as utils
import argparse
from datalayer.hash_algorithm.tlsh_algorithm import TLSHHashAlgorithm
if __name__ == '__main__':   
    parser = argparse.ArgumentParser()

    # get log level from command line
    parser.add_argument('-log', '--loglevel', choices=["debug", "info", "warning", "error", "critical"], default='warning', help="Provide logging level (default=warning)")
    args = parser.parse_args()

    utils.configure_logging(args.loglevel.upper())
    hash1 = "T1BF81A292E336D1F68224D4A4C751A2B3BB353CA9C2103BA69FA4C7908761B50F22E301"
    hash2 = "T12B81E2134758C0E3CA097B381202C62AC793B46686CD9E2E8F9190EC89C537B5E7AF4C" 
    hash3 = "T10381E956C26225F2DAD9D5C2C5C1A337FAF3708A25012B8A1EACDAC00B37D557E0E714"
    hash4 = "T1DF8174A9C2A506F9C6FFC292D6816333FEF1B845C419121A0F91CF5359B5B21FA3A304"
    hash5 = "T1DF8174A9C2A506F9C6FFC292D6816333FEF1B845C419121A0F91CF5359B5B21FA3A305"
    hash6 = "T1DF8174A9C2A506FC122292D644816333FEF1B845C419121A0F91CF5359B5B21FA3A305"
    hash7 = "T10381E956C26225F2DAD9D097B381202C62AC793B37082B8A1EACDAC00B37D557E0E714"
    
    _hashes = [hash1, hash2,\
                hash3, hash4,\
                hash5, hash6, hash7
                ]

    # RadixHash tree object
    tree = RadixHash(TLSHHashAlgorithm)
 
    # Construct radix hash tree randomly
    _list = random.sample(_hashes, 5)
    for _hash in _list:
        print(f"[*] Inserting \"{_hash}\" in the radix hash tree ... ")
        tree.insert(HashNode(_hash, TLSHHashAlgorithm))
    
    print("[*] Printing the radix hash tree ...")
    tree.print_tree()
 
    
    # Search for different keys
    _list = random.sample(_hashes, 3)
    for _hash in _list:
        print(f"[*] Searching \"{_hash}\" in the radix hash tree ... ", end='')
        found, _ = tree.search(_hash)
        if found: 
            print("Yep! Contained");
        else:
            print("Not contained :(");
    
    # save it
    filename = "myRadixTree.radix_tree"
    print(f"[*] Saving radix tree to {filename} ...")
    tree.dump(filename)

    # restore it
    print(f"[*] Restoring radix tree from {filename} and printing it again ...")
    myTree = RadixHash.load(filename)
    tree.print_tree(only_hashes=True)

