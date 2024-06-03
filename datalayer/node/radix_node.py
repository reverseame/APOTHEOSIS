import pickle
import logging

from datalayer.node.hash_node import HashNode

from common.errors import NodeAlreadyExistsError

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

# code adapted from https://github.com/TheAlgorithms/Python/blob/master/data_structures/trie/radix_tree.py
class RadixHashNode:

    def __init__(self, prefix: str = "", hash_node: HashNode=None) -> None:
        # Mapping from the first character of the prefix of the node
        self._nodes: dict[str, RadixNode] = {}

        # A node will be a leaf if the tree contains a HashNode pointer not None
        self._hash_node = hash_node
        self._prefix = prefix

    def _match(self, word: str) -> tuple[str, str, str]:
        """Computes the common substring of the prefix of the node and a word.
        Returns a the common substring, remaining prefix, and remaining word

        Arguments
        word    -- word to compare
        """
        
        logger.debug(f"Searching \"{word}\" match in prefix \"{self._prefix}\"...")
        x = 0
        for q, w in zip(self._prefix, word):
            if q != w:
                break

            x += 1

        common_substr    = self._prefix[:x]
        remaining_prefix = self._prefix[x:]
        remaining_word   = word[x:]
        logger.debug(f"Match found in {x}: {common_substr}, {remaining_prefix}, {remaining_word}")
        return common_substr, remaining_prefix, remaining_word

    def insert(self, word: str, hash_node: HashNode):
        """Inserts a hash node into the radix hash tree

        Arguments:
        word        -- hash word to insert
        hash_node   -- hash node to insert
        """
        
        logger.debug(f"Inserting node \"{hash_node.get_id()}\" (current word: {word})")
        # Case 1: If the word is the prefix of the node
        # Solution: We set the current node as leaf
        if self._prefix == word and self._hash_node is None:
            logger.debug(f"Current node set as leaf, prefix=\"{self._prefix}\"")
            self._hash_node = hash_node 
            # in our case (with hashes), this should never occurr if we always the same hash length

        # Case 2: The node has no edges that have a prefix to the word
        # Solution: We create an edge from the current node to a new one
        # containing the word
        elif word[0] not in self._nodes:
            logger.debug(f"Current node has no edges with this prefix, create a new node with prefix \"{word}\"")
            self._nodes[word[0]] = RadixHashNode(prefix=word, hash_node=hash_node)

        else:
            incoming_node = self._nodes[word[0]]
            matching_string, remaining_prefix, remaining_word = incoming_node._match(word)

            # Case 3: The node prefix is equal to the matching
            # Solution: We insert remaining word on the next node
            if remaining_prefix == "":
                if remaining_word == "": # if both are "", we already have the hash in our radix tree
                    raise NodeAlreadyExistsError
                logger.debug(f"Remaining prefix is empty, inserting now \"{remaining_word}\"")
                self._nodes[matching_string[0]].insert(remaining_word, hash_node)

            # Case 4: The word is greater equal to the matching
            # Solution: Create a node in between both nodes, change
            # prefixes and add the new node for the remaining word
            else:
                logger.debug(f"{word} is >= to the matching, create a new node, change prefixes, and add \"{remaining_word}\"")
                incoming_node._prefix = remaining_prefix
                logger.debug(f"Prefix changed to \"{incoming_node._prefix}\"")

                aux_node = self._nodes[matching_string[0]]

                self._nodes[matching_string[0]] = RadixHashNode(matching_string, None)
                self._nodes[matching_string[0]]._nodes[remaining_prefix[0]] = aux_node

                if remaining_word == "":
                    logger.debug(f"Associating \"{hash_node.get_id()}\" in this node ...")
                    # check if already exists and raise exception if necessary
                    self._nodes[matching_string[0]]._hash_node = hash_node
                else:
                    logger.debug(f"Inserting \"{remaining_word}\" ...")
                    self._nodes[matching_string[0]].insert(remaining_word, hash_node)
    
    def delete(self, word: str) -> HashNode:
        """Deletes a hash word from the radix hash tree, if it exists.
        Returns the HashNode matching with the word if it was found and deleted, None otherwise

        Arguments:
        word    -- word to delete
        """
        incoming_node = self._nodes.get(word[0], None)
        if not incoming_node:
            return None
        else:
            matching_string, remaining_prefix, remaining_word = incoming_node._match(word)
            # If there is remaining prefix, the word can't be on the tree
            if remaining_prefix != "":
                return None
            # We have word remaining so we check the next node
            elif remaining_word != "":
                return incoming_node.delete(remaining_word)
            else:
                # If it is not a leaf, we don't have to delete
                if not incoming_node._hash_node:
                    return None
                else:
                    hash_node = incoming_node._hash_node
                    # We delete the nodes if no edges go from it
                    if len(incoming_node._nodes) == 0:
                        del self._nodes[word[0]]
                        # We merge the current node with its only child
                        if len(self._nodes) == 1 and not self._hash_node:
                            merging_node = next(iter(self._nodes.values()))
                            self._hash_node = merging_node._hash_node
                            self._prefix += merging_node._prefix
                            self._nodes = merging_node._nodes
                    # If there is more than 1 edge, we just mark it as non-leaf
                    elif len(incoming_node._nodes) > 1:
                        incoming_node._hash_node = None
                    # If there is 1 edge, we merge it with its child
                    else:
                        merging_node = next(iter(incoming_node._nodes.values()))
                        incoming_node._hash_node = merging_node._hash_node
                        incoming_node._prefix += merging_node._prefix
                        incoming_node._nodes = merging_node._nodes

                    return hash_node

    def print_tree(self, height: int=0, only_hashes: bool=False):
        """Prints the radix hash tree

        Arguments:
        height      -- height of the printed node
        only_hashes -- bool flag to indicate if we only need to print hashes contained in the tree        
        """
        if self._prefix != "":
            if only_hashes:
                if self._hash_node:
                    print("-" * height, f" {self._hash_node.get_id()}")
            else:
                print("-" * height, self._prefix, f"  ({self._hash_node.get_id()})" if self._hash_node else "")

        for value in self._nodes.values():
            value.print_tree(height + 1, only_hashes)
    
    def search(self, word: str, hash_node: HashNode) -> (bool, HashNode):
        """Returns True and the associated hash node if the word is on the radix hash tree, (False, None) otherwise

        Arguments:
        word        -- word to check
        hash_node   -- incoming hash node (necessary by the recursion)
        """
        incoming_node = self._nodes.get(word[0], None)
        if not incoming_node:
            return False, None
        else:
            matching_string, remaining_prefix, remaining_word = incoming_node._match(word)
            # if there is remaining prefix, the word can't be on the tree
            if remaining_prefix != "":
                return False, None
            # this applies when the word and the prefix are equal
            elif remaining_word == "":
                return (incoming_node._hash_node is not None), incoming_node._hash_node
            # We have word remaining so we check the next node
            else:
                return incoming_node.search(remaining_word, hash_node)

    def __str__(self):
        _str = f"RadixHashNode ID: '{self._prefix}' "
        if self._hash_node:
            _str += "\"" + str(self._hash_node.get_id()) + "\", Neighbors: " + self._hash_node.print_neighbors()
        return _str
    
    def __repr__(self): # for printing while iterating RadixHashNode data structures
        return "<" + str(self) + ">"
