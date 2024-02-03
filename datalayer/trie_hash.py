import pickle
import logging

from datalayer.node.node_hash import HashNode
from datalayer.node.trie_node import TrieHashNode
from datalayer.errors import NodeAlreadyExistsError

__author__ = "Daniel Huici Meseguer and Ricardo J. Rodríguez"
__copyright__ = "Copyright 2024"
__credits__ = ["Daniel Huici Meseguer", "Ricardo J. Rodríguez"]
__license__ = "GPL"
__version__ = "0.3"
__maintainer__ = "Daniel Huici"
__email__ = "reverseame@unizar.es"
__status__ = "Development"

logger = logging.getLogger(__name__)
logging.getLogger('pickle').setLevel(logging.WARNING)

# TrieHash data structure class
# code adapted from https://www.geeksforgeeks.org/trie-insert-and-search/
class TrieHash:
    def __init__(self, distance_algorithm):
        """Default constructor.

        Arguments:
        distance_algorithm  -- distance algorithm that creates the hashes stored in the trie
        """
        self._distance_algorithm = distance_algorithm
        self._alphalen = distance_algorithm.get_max_hash_alphalen()
        self._root = self._new_trie_node()

    def dump(self, file):
        """Saves trie structure to permanent storage.

        Arguments:
        file    -- filename to save
        """

        with open(file, "wb") as f:
            pickle.dump(self, f, protocol=pickle.DEFAULT_PROTOCOL)

    @classmethod
    def load(cls, file):
        """Restores trie structure from permanent storage.

        Arguments:
        file    -- filename to load
        """
        with open(file, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected an instance of {cls.__name__}, but got {type(obj).__name__}")
        return obj
 
    def get_alphalen(self):
        """Getter for _alphalen."""
        return self._alphalen
    
    def get_distance_algorithm(self):
        """Getter for _distance_algorithm."""
        return self._distance_algorithm

    def _new_trie_node(self, hash_node: HashNode=None):
        """Returns a new trie node (hash_node initialized to None).
        
        Arguments:
        hash_node   -- hash node associated to the trie node
        """
        return TrieHashNode(self._alphalen, hash_node)
 
    def _char_to_index(self, ch):
        """Converts key current character into index, considering the trie alphabet.
        
        Arguments:
        ch  -- char to convert
        """
        # consider digits + upper case + lower case alphabet
        if '0' <= ch and ch <= '9':
            return ord(ch) - ord('0')
        
        _shift = ord('9') - ord('0') + 1 
        if 'A' <= ch and ch <= 'Z':
            return ord(ch) - ord('A') + _shift 
        
        _shift += ord('Z') - ord('A') + 1 
        return ord(ch) - ord('a') + _shift
 
    def insert(self, hash_node: HashNode):
        """Inserts a new hash node (identified by its id) in the trie.
        If the hash node is not present, it inserts the hash node into the trie and marks the leaf node
        Otherwise, it raises NodeAlreadyExistsError.
        
        Arguments:
        hash_node   -- new node to add
        """
       
        logger.info(f"Inserting \"{hash_node.get_id()} in the trie ...\"")
        p_crawl = self._root
        key = hash_node.get_id()
        for level in range(len(key)):
            index = self._char_to_index(key[level])
 
            # if current character is not present
            if not p_crawl._children[index]:
                logger.debug(f"Creating new node in children {index} (char is '{key[level]}')")
                p_crawl._children[index] = self._new_trie_node()
            p_crawl = p_crawl._children[index]

        logger.debug(f"Leaf node reached for \"{key}\"")
        if p_crawl._hash_node is not None:
            raise NodeAlreadyExistsError
        else:
            # mark last node as leaf storing hash_node
            p_crawl._hash_node = hash_node
 
    def search(self, key: str):
        """Searchs the key in the trie. Returns True and the hash node if key exists in the trie, 
        False and None otherwise.
        
        Arguments:
        key -- key to search
        """
        logger.info(f"Searching \"{key}\" in the trie ...")
         
        p_crawl = self._root
        length = len(key)
        for level in range(length):
            index = self._char_to_index(key[level])
            if not p_crawl._children[index]:
                logger.debug(f"\"{key}\" not found at level {level}, search is finished (char is '{key[level]}')")
                return False, None
            p_crawl = p_crawl._children[index]
        
        logger.debug(f"Leaf node for \"{key}\" reached (found? {p_crawl._hash_node is not None})")
        return (p_crawl._hash_node is not None), p_crawl._hash_node
 
# unit test
# run this as "python3 -m datalayer.trie_hash"
import random
from datalayer.hash_algorithm.tlsh_algorithm import TLSHHashAlgorithm
if __name__ == '__main__':
    hash1 = "T1BF81A292E336D1F68224D4A4C751A2B3BB353CA9C2103BA69FA4C7908761B50F22E301"
    hash2 = "T12B81E2134758C0E3CA097B381202C62AC793B46686CD9E2E8F9190EC89C537B5E7AF4C" 
    hash3 = "T10381E956C26225F2DAD9D5C2C5C1A337FAF3708A25012B8A1EACDAC00B37D557E0E714"
    hash4 = "T1DF8174A9C2A506F9C6FFC292D6816333FEF1B845C419121A0F91CF5359B5B21FA3A304"
    hash5 = "T1DF8174A9C2A506F9C6FFC292D6816333FEF1B845C419121A0F91CF5359B5B21FA3A305"
    
    _hashes = [hash1, hash2,\
                hash3, hash4,\
                hash5
                ]

    # TrieHash object
    t = TrieHash(TLSHHashAlgorithm)
 
    # Construct trie randomly
    _list = random.sample(_hashes, 3)
    for _hash in _list:
        print(f"Inserting \"{_hash}\" in the trie ... ")
        t.insert(HashNode(_hash, TLSHHashAlgorithm))
 
    # Search for different keys
    _list = random.sample(_hashes, 3)
    for _hash in _list:
        print(f"Searching \"{_hash}\" in the trie ... ", end='')
        found, _ = t.search(_hash)
        if found: 
            print("Yep! Contained");
        else:
            print("Not contained :(");

