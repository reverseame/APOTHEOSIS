# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
import zlib
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

# for compressed dumping
import gzip as gz
import io

from common.constants import * 

from datalayer.db_manager import DBManager

from datalayer.radix_hash import RadixHash
from datalayer.hnsw import HNSW
from datalayer.hash_algorithm.hash_algorithm import HashAlgorithm

# custom exceptions
from common.errors import NodeNotFoundError
from common.errors import NodeAlreadyExistsError

from common.errors import ApotheosisUnmatchDistanceAlgorithmError
from common.errors import ApotheosisIsEmptyError
from common.errors import ApotFileFormatUnsupportedError
from common.errors import ApotFileReadError

# preferred file extension
PREFERRED_FILEEXT = ".apo"

class Apotheosis(ABC):
    def create_empty(self, M=0, ef=0, Mmax=0, Mmax0=0,\
                    distance_algorithm=None,\
                    heuristic=False, extend_candidates=True, keep_pruned_conns=True,\
                    beer_factor: float=0):
        # construct both data structures (a HNSW and a radix tree for all nodes -- will contain @WinModuleHashNode)
        self._HNSW = HNSW(M=M, ef=ef, Mmax=Mmax, Mmax0=Mmax0, distance_algorithm=distance_algorithm,\
                                heuristic=heuristic, extend_candidates=extend_candidates, keep_pruned_conns=keep_pruned_conns,\
                                beer_factor=beer_factor)
        self._distance_algorithm = distance_algorithm
        # radix hash tree for all nodes (of @WinModuleHashNode)
        self._radix = RadixHash(distance_algorithm)

    @classmethod
    def _check_compression(cls, f):
        """Checks if the file is compresed.
        If compressed, it decompress and returns it. Otherwise, it returns the same f
        """
        
        logger.info(f"Checking if {f.name} is compressed ...")
        magic = b'\x1f\x8b\x08' # magic bytes of gzip file
        compressed = False
        with open(f.name, 'rb') as fp:
            start_of_file = fp.read(1024)
            fp.seek(0)
            compressed = start_of_file.startswith(magic)
        
        # if compressed, load the appropriate file
        if not compressed:
            logger.debug(f"Not compressed. Desearializing it directly ...")
        else:
            logger.debug(f"Compressed. Decompressing and deserializing ...")
            f = gz.GzipFile(f.name)
        return f

    @abstractmethod
    def _load_node_from_fp(cls, f, pageid_to_node: dict,  
                                with_layer:bool=False, algorithm: HashAlgorithm=None, db_manager=None):
        raise NotImplementedError

    @classmethod
    def _assert_header(cls, byte_data: bytearray):
        """Checks header file and returns CRC32 of HNSW cfg, enter point, and nodes read from the byte data array.

        Arguments:
        byte_data   -- byte data to process
        """
        logger.debug(f"Checking header: {byte_data}")
        if len(byte_data) != HEADER_SIZE:
            raise ApotFileFormatUnsupportedError
        # check magic
        magic = byte_data[0:len(MAGIC_BYTES)]
        if magic != MAGIC_BYTES:
            raise ApotFileFormatUnsupportedError
        # check version
        version = byte_data[len(MAGIC_BYTES): len(MAGIC_BYTES) + C_SIZE]
        if version != VERSIONFILE:
            raise ApotFileFormatUnsupportedError
        
        idx = len(MAGIC_BYTES) + C_SIZE
        CRC32_bcfg      = int.from_bytes(byte_data[idx:idx + I_SIZE], byteorder=BYTE_ORDER)
        CRC32_bep       = int.from_bytes(byte_data[idx + I_SIZE:idx + I_SIZE*2], byteorder=BYTE_ORDER)
        CRC32_bnodes    = int.from_bytes(byte_data[idx + I_SIZE*2:idx + I_SIZE*3], byteorder=BYTE_ORDER)

        return CRC32_bcfg, CRC32_bep, CRC32_bnodes
        
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
    
    def _serialize_apoth_node(self, node, with_layer: bool=False) -> bytearray:
        """Returns a byte array representing node.

        Arguments:
        node        -- node to serialize
        with_layer  -- bool flag to indicate if we serialize also max layer of the node
        """
        logger.debug(f"Serializing \"{node.get_id()}\" ...")
        max_layer   = node.get_max_layer()
        logger.debug(f"Node at L{max_layer}")
        # convert integer to bytes (needs to follow BYTE_ORDER)
        bstr = node.internal_serialize()
        if with_layer:                                                # <N_LAYER> (only ep)
            bstr += max_layer.to_bytes(I_SIZE, byteorder=BYTE_ORDER)  # sizes in constants  
        
        neighs_list = node.get_neighbors()
        # print first the number of layers
        bstr += len(neighs_list).to_bytes(I_SIZE, byteorder=BYTE_ORDER)     # <N_HOODS>
        logger.debug(f"Neighborhoods len: {len(neighs_list)}")
        # iterate now in neighbors
        for layer, neighs_set in enumerate(neighs_list): 
            logger.debug(f"Processing L{layer} ...")
            bstr += layer.to_bytes(I_SIZE, byteorder=BYTE_ORDER) +\
                     len(neighs_set).to_bytes(I_SIZE, byteorder=BYTE_ORDER) # <N_LAYER> <N_NEIGS>
            # get internal data of the neighs
            bstr += b''.join([node.internal_serialize() for node in neighs_set])
       
        logger.debug(f"Node serialized: {bstr}")
        return bstr

    def dump(self, filename: str, compress: bool=True):
        """Saves Apotheosis structure to permanent storage.

        Arguments:
        filename    -- filename to save
        compress    -- bool flag to compress the output file
        """

        logger.info(f"Saving Apotheosis structure to \"{filename}\" (compressed? {compress}) ...")

        logger.debug("Serializing HNSW configuration ... ") 
        bcfg = self._HNSW.serialize_cfg() 
        CRC32_bcfg = zlib.crc32(bcfg) & 0xffffffff
        logger.debug("Serializing enter point ... ") 
        ep = self._HNSW.get_enter_point()
        bep = self._serialize_apoth_node(ep, with_layer=True)
        # guarantees compatibility -- https://stackoverflow.com/questions/30092226/
        CRC32_bep = zlib.crc32(bep) & 0xffffffff
        # now, iterate on layers, printing each node and its neighbors
        bnodes = bytes() 
        # write first the number of layers
        bnodes += len(self._HNSW._nodes).to_bytes(I_SIZE, byteorder=BYTE_ORDER)
        for layer, node_list in self._HNSW._nodes.items():
            # XXX we always double the relationships between neighbors because we write their
            # page ids twice (one per relation) -- otherwise, I don't know how to break the recursion here
            if ep.get_max_layer() == layer: # avoid repeated storing of enter point 
                node_list.remove(ep)     
            logger.debug(f"Length of nodes to serialize at L{layer}: {len(node_list)} ...")
            
            # write current layer number and neighbors here 
            bnodes += layer.to_bytes(I_SIZE, byteorder=BYTE_ORDER)
            bnodes += len(node_list).to_bytes(I_SIZE, byteorder=BYTE_ORDER)

            for node in node_list:
                logger.debug(f"Serializing a new node ...")
                bnodes += self._serialize_apoth_node(node)

        # we add again the ep, as it was removed before (when doing "node_list.remove(ep)")
        self._HNSW._insert_node(ep)

        CRC32_bnodes = zlib.crc32(bnodes) & 0xffffffff
        logger.debug(f"CRC32s computed: bcfg={hex(CRC32_bcfg)}, bep={hex(CRC32_bep)}, bnodes={hex(CRC32_bnodes)}...")
        logger.debug("All data from objects serialized. Dumping to file now ...")
         
        # build header as magic + version + CRC32 of each part
        magic = MAGIC_BYTES
        version = VERSIONFILE
        header = magic + version \
                       + CRC32_bcfg.to_bytes(I_SIZE, byteorder=BYTE_ORDER)\
                       + CRC32_bep.to_bytes(I_SIZE, byteorder=BYTE_ORDER)\
                       + CRC32_bnodes.to_bytes(I_SIZE, byteorder=BYTE_ORDER)
        
        # create now the file
        if compress:
            f = io.BytesIO()
        else:
            f = open(filename, "wb")
        
        try: # see FILEFORMAT.md for details
            f.write(header)
            # first, HNSW configuration
            f.write(bcfg)
            # then, enter point
            f.write(bep)
            # finally, nodes
            f.write(bnodes)
            f.write(EOF) # eof
        except Exception as e: 
            pass # there is nothing you can do
        
        # compress the file 
        if compress:
            compressed_data = gz.compress(f.getvalue())
            with open(filename, "wb") as fp:
                fp.write(compressed_data)
                fp.close()
            logger.debug(f"Compressing memory file and saving it to {filename} ... done!")
        
        f.close() # done!
        
        return

    @abstractmethod
    def load(cls, filename:str=None, distance_algorithm=None):
        """Restores Apotheosis structure from permanent storage.
        
        Arguments:
        filename            -- filename to load
        distance_algorithm  -- distance algorithm to check in the file
        """
        raise NotImplementedError

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

    def search_exact_match_only(self, hash_value):
        """Returns an exact match search of hash value and a bool found flag.
        
        Arguments:
        hash_value  -- hash value to search (in the radix tree only)
        """
        
        found, node = self._radix.search(hash_value)
        logger.info("Trying exact match for \"{hash_value}\" ... found? {found}")
        return found, node

    def knn_search(self, query=None, k:int=0, ef=0, hashid=0):
        """If query is present in the Apotheosis structure, returns True, the node found, and the K nearest neighbors to query. 
        Otherwise, returns False, None, and the approximate K nearest neighbors to query.
        It raises the following exceptions:
            * ApotheosisUnmatchDistanceAlgorithmError if the distance algorithm of the new node is distinct than 
              the distance algorithm associated to the HNSW structure.
            * ApotheosisIsEmptyError if the HNSW structure is empty

        Arguments:
        query   -- base node
        k       -- number of nearest neighbors to query node to return
        ef      -- exploration factor (search recall)
        hashid  -- hash str to search
        """
        if hashid != 0:
            # create node and make the search again...
            query = WinModuleHashNode(hashid, self.get_distance_algorithm()) 
        
        self._sanity_checks(query)
        
        logger.info(f"Performing a KNN search for \"{query.get_id()}\" (k={k}, ef={ef})")
        exact, node = self.search_exact_match_only(query.get_id())
        if exact: # get k-nn at layer 0, using HNSW structure
            # as node exists, this call is safe
            logger.debug(f"Node \"{query.get_id()}\" found in the radix tree! Recovering now its neighbors from HNSW ... ")
            knn_dict = self._HNSW.get_knn_node_at_layer(node, k, layer=0) 
        else: # get approximate k-nns with HNSW search
            logger.debug(f"Node \"{query.get_id()}\" NOT found in the radix tree! Recovering now its approximate neighbors ... ")
            knn_dict = self._HNSW.aknn_search(query, k, ef)    # log N, see Section 4.2.1 in MY-TPAMI-20
            node = None

        return exact, node, knn_dict

    def threshold_search(self, query, threshold, n_hops):
        """Performs a threshold search to retrieve nodes that satisfy a certain similarity threshold using the HNSW structure.
        If query is present in the Apotheosis structure, returns True, the node found, and the list of nearest neighbor nodes 
        to query that satisfy the specified similarity threshold.
        Otherwise, returns False, None, and the approximate K nearest neighbors to query.
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
        exact, node = self.search_exact_match_only(query.get_id())
        if exact: # get k-nn at layer 0, using HNSW structure
            # as node exists, this is safe
            logger.debug(f"Node \"{query.get_id()}\" found in the radix tree! Recovering now its neighbors ... ")
            knn_dict = self._HNSW.get_thresholdnn_at_node(query, threshold) 
        else: # get approximate k-nns with HNSW search
            logger.debug(f"Node \"{query.get_id()}\" NOT found in the radix tree! Recovering now its approximate neighbors ... ")
            knn_dict = self._HNSW.threshold_search(query, threshold, n_hops)
            node = None

        return exact, node, knn_dict

    def draw_hashes_subset(self, hash_set: set, filename: str, show_distance: bool=True, format="pdf", cluster:bool=False):
        """Creates a graph figure per level of the HNSW structure and saves it to a filename file, 
        but only considering hash values in hash_set.

        Arguments:
        hash_set        -- set of nodes to draw
        filename        -- filename to create (with extension)
        show_distance   -- to show the distance metric in the edges (default is True)
        format          -- matplotlib plt.savefig(..., format=format) (default is "pdf")
        cluster         -- bool flag to draw also the structure in cluster mode (considering modules)
        """
        
        logger.info(f"Drawing to {filename} (subset: {hash_set} with cluster? {cluster}) ...")
        self._HNSW.draw(filename, show_distance=show_distance, format=format, hash_subset=hash_set, cluster=cluster)

    def draw(self, filename: str, show_distance: bool=True, format="pdf", cluster: bool=False, threshold: float=0.0):
        """Creates a graph figure per level of the HNSW structure and saves it to a filename file.

        Arguments:
        filename        -- filename to create (with extension)
        show_distance   -- to show the distance metric in the edges (default is True)
        format          -- matplotlib plt.savefig(..., format=format) (default is "pdf")
        cluster         -- bool flag to draw also the structure in cluster mode (considering modules)
        """
        logger.info(f"Drawing to {filename} (with cluster? {cluster}) ...")
        self._HNSW.draw(filename, show_distance=show_distance, format=format, cluster=cluster, threshold=threshold)

    # to support ==, now the object is not unhasheable (cannot be stored in sets or dicts)
    def __eq__(self, other):
        """Returns True if this object and other are the same, False otherwise.

        Arguments:
        other   -- HNSW object to check
        """
        if type(self) != type(other):
            return False

        return self._HNSW == other._HNSW

