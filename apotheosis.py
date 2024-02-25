# -*- coding: utf-8 -*-
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

class Apotheosis:
    
    def __init__(self, M=0, ef=0, Mmax=0, Mmax0=0,\
                    distance_algorithm=None,\
                    heuristic=False, extend_candidates=True, keep_pruned_conns=True,\
                    beer_factor: float=0,\
                    filename=None, db_manager=None):
        """Default constructor."""
        if filename == None:
            # construct both data structures (a HNSW and a radix tree for all nodes -- will contain @WinModuleHashNode)
            self._HNSW = HNSW(M=M, ef=ef, Mmax=Mmax, Mmax0=Mmax0, distance_algorithm=distance_algorithm,\
                                heuristic=heuristic, extend_candidates=extend_candidates, keep_pruned_conns=keep_pruned_conns,\
                                beer_factor=beer_factor)
            self._distance_algorithm = distance_algorithm
            # radix hash tree for all nodes (of @WinModuleHashNode)
            self._radix = RadixHash(distance_algorithm)
        else:
            # open the file and load structures from filename
            with open(filename, "rb") as f:
                # check if file is compressed and do stuff, if necessary
                f = Apotheosis._check_compression(f)
                # read the header and process
                data = f.read(HEADER_SIZE)
                # check header (file format and version match)
                rCRC32_bcfg, rCRC32_bep, rCRC32_bnodes = Apotheosis._assert_header(data)
                logger.debug(f"CRCs read: bcfg={hex(rCRC32_bcfg)}, bep={hex(rCRC32_bep)}, bnodes={hex(rCRC32_bnodes)}")
                # check HNSW cfg and load it if no error
                data = f.read(CFG_SIZE)
                CRC32_bcfg = zlib.crc32(data) & 0xffffffff
                if CRC32_bcfg != rCRC32_bcfg:
                    raise ApotFileReadError(f"CRC32 {hex(CRC32_bcfg)} of HNSW configuration does not match (should be {hex(rCRC32_bcfg)})")
                self._HNSW = HNSW.load_cfg_from_bytes(data)
               
                if self._HNSW.get_distance_algorithm() != distance_algorithm:
                    raise ApotheosisUnmatchDistanceAlgorithmError

                self._distance_algorithm = self._HNSW.get_distance_algorithm()
                pageid_to_node = {}
                pageid_neighs = {}
                logger.debug(f"Reading enter point from file \"{filename}\" ...")
                # now, process enter point
                ep, pageid_to_node, pageid_neighs, CRC32_bep, _ = \
                        Apotheosis._load_node_from_fp(f, pageid_to_node, pageid_neighs, with_layer=True,
                                                        algorithm=distance_algorithm, db_manager=db_manager)
                if CRC32_bep != rCRC32_bep:
                    raise ApotFileReadError(f"CRC32 {hex(CRC32_bep)} of enter point does not match (should be {hex(rCRC32_bep)})")
               
                breakpoint()
                self._HNSW._enter_point  = ep 
                self._HNSW._insert_node(ep) # internal, add the node to nodes dict
                # finally, process each node in each layer
                n_layers = f.read(I_SIZE)
                bnodes = n_layers
                n_layers = int.from_bytes(n_layers, byteorder=BYTE_ORDER)
                logger.debug(f"Reading {n_layers} layers ...")
                for _layer in range(0, n_layers):
                    # read the layer number
                    layer = f.read(I_SIZE)
                    bnodes += layer
                    layer = int.from_bytes(layer, byteorder=BYTE_ORDER)
                    # read the number of nodes in this layer
                    neighs_to_read = f.read(I_SIZE)
                    bnodes += neighs_to_read
                    neighs_to_read = int.from_bytes(neighs_to_read, byteorder=BYTE_ORDER)
                    logger.debug(f"Reading {neighs_to_read} nodes at L{layer} ...")
                    for idx in range(0, neighs_to_read):
                        new_node, pageid_to_node, current_pageid_neighs, _, bnode = \
                            Apotheosis._load_node_from_fp(f, pageid_to_node, pageid_neighs, 
                                                        algorithm=distance_algorithm, db_manager=db_manager)
                        new_node.set_max_layer(layer)
                        self._HNSW._insert_node(new_node) # internal, add the node to nodes dict
                        pageid_neighs.update(current_pageid_neighs)
                        bnodes += bnode
                    
                CRC32_bnodes = zlib.crc32(bnodes) & 0xffffffff
                logger.debug(f"Nodes loaded correctly. CRC32 computed: {hex(CRC32_bnodes)}")
                if CRC32_bnodes != rCRC32_bnodes:
                    raise ApotFileReadError(f"CRC32 {hex(CRC32_bnodes)} of nodes does not match (should be {hex(rCRC32_bnodes)})")
            # all done here, except we need to link neighbors...
            for pageid in pageid_neighs:
                # search the node -- this search should always return something
                try:
                    node = pageid_to_node[pageid]
                except Exception as e:
                    raise ApotFileReadError(f"Node with pageid {pageid} not found. Is this code correct?")
                
                neighs = pageid_neighs[pageid]
                for layer in neighs:
                    logger.debug(f"Recreating nodes at L{layer} ...")
                    neighs_at_layer = neighs[layer]
                    for neigh in neighs_at_layer:
                        logger.debug(f"Recreating node with pageid {neigh} at L{layer} ...")
                        # search the node -- this search should always return something
                        try:
                            neigh_node = pageid_to_node[neigh]
                        except Exception as e:
                            raise ApotFileReadError(f"Node with pageid {neigh} not found. Is this code correct?")
                        # add the link between them
                        node.add_neighbor(layer, neigh_node)
                        # (the other link will be set later, when processing the neigh as node)
            
            # recreate radix tree from HNSW (we can do it also in the loop above)
            self._radix = RadixHash(self._distance_algorithm, self._HNSW)

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

    @classmethod
    def _load_node_from_fp(cls, f, pageid_to_node: dict, pageid_neighs: dict, 
                                with_layer:bool=False, algorithm: HashAlgorithm=None, db_manager=None):
        """Loads a node from a file pointer f.
        It is necessary to have a db_manager to load an Apotheasis file from disk
        (we only keep page ids and their relationships, nothing else).

        Arguments:
        f               -- file pointer to read from
        pageid_to_node  -- dict to map page ids to WinModuleHashNode (necessary for rebuilding indexes)
        pageid_neighs   -- dict to map page ids to neighbors page ids, per layer level (necessary for rebuilding indexes)
        with_layer      -- bool flag to indicate if we need to read the layer for this node (default False)
        algorithm       -- associated distance algorithm
        db_manager      -- db_manager to handle connections to DB (optional)
        """
        logger.debug("Loading a new node from file pointer ...")
       
        page_id     = f.read(I_SIZE)
        bnode       = page_id
        max_layer   = '(no layer)' 
        if with_layer:
            max_layer   = f.read(I_SIZE)
            bnode      += max_layer
            max_layer   = int.from_bytes(max_layer, byteorder=BYTE_ORDER)
        
        logger.debug(f"Read page id: {page_id}, layer: {max_layer} ...")
        page_id     = int.from_bytes(page_id, byteorder=BYTE_ORDER)
        # read neighborhoods
        nhoods      = f.read(I_SIZE)
        logger.debug(f"Read neighborhoods: {nhoods} ...")
        bnode      += nhoods
        nhoods      = int.from_bytes(nhoods, byteorder=BYTE_ORDER)
        logger.debug(f"Node {page_id} with {nhoods} neighborhoods, starts processing ...")
        neighs_page_id = {} 
        layer = 0
        # process each neighborhood, per layer and neighbors in that layer
        for nhood in range(0, nhoods):
            logger.debug(f"Processing neighborhood {nhood} ...")
            layer   = f.read(I_SIZE)
            neighs  = f.read(I_SIZE)
            logger.debug(f"Read {neighs} neighbors and layer {layer} ...")
            bnode  += layer + neighs
            layer   = int.from_bytes(layer, byteorder=BYTE_ORDER)
            neighs  = int.from_bytes(neighs, byteorder=BYTE_ORDER)
            neighs_page_id[layer] = []
            # get now the neighs page id at this layer 
            for idx_neigh in range(0, neighs):
                neigh_page_id = f.read(I_SIZE)
                logger.debug(f"Read neigh page id: {neigh_page_id} ...")
                bnode        += neigh_page_id
                neighs_page_id[layer].append(int.from_bytes(neigh_page_id, byteorder=BYTE_ORDER))
            logger.debug(f"Processed {neighs} at L{layer} ({neighs_page_id})")

        CRC32_bnode = zlib.crc32(bnode) & 0xffffffff
        logger.debug(f"New node with {page_id} at L{layer} successfully read. Neighbors page ids: {neighs_page_id}. Computed CRC32: {hex(CRC32_bnode)}")

        # retrieve the specific page id information from database and get a WinModuleHashNode
        logger.debug("Recovering data now from DB, if necessary ...")
        new_node        = None
        pageid_neighs   = {} 
        if db_manager:
            if pageid_to_node.get(page_id) is None:
                new_node = db_manager.get_winmodule_data_by_pageid(page_id=page_id, algorithm=algorithm)
                if algorithm == TLSHHashAlgorithm:
                    new_node._id = new_node._page.hashTLSH
                elif algorithm == SSDEEPHashAlgorithm:
                    new_node._id = new_node._page.hashSSDEEP
                else:
                    raise ApotFileFormatUnsupportedError
                if with_layer:
                    new_node.set_max_layer(max_layer)
                # store it for next iterations
                pageid_to_node[page_id] = new_node
            else:
                #breakpoint()
                new_node = pageid_to_node[page_id]
            logger.debug(f"Initial data set to new node: \"{new_node.get_id()}\" at L{max_layer}")

            # get now the neighboors
            if pageid_neighs.get(page_id) is None:
                pageid_neighs[page_id] = {}
            for layer, neighs_list in neighs_page_id.items():
                if pageid_neighs[page_id].get(layer) is None:
                    pageid_neighs[page_id][layer] = set()
                pageid_neighs[page_id][layer].update(neighs_list)
        else:
            logger.debug("No db_manager was given, skipping data retrieval from DB ...")

        return new_node, pageid_to_node, pageid_neighs, CRC32_bnode, bnode 

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
        page_id     = node.get_internal_page_id()
        logger.debug(f"Node at L{max_layer} with page_id={page_id}")
        # convert integer to bytes (needs to follow BYTE_ORDER)
        bstr = page_id.to_bytes(I_SIZE, byteorder=BYTE_ORDER)               # <page-id> 
        if with_layer:                                                      # <N_LAYER> (only ep)
            bstr += max_layer.to_bytes(I_SIZE, byteorder=BYTE_ORDER)  # sizes in constants  
        
        neighs_list = node.get_neighbors()
        # print first the number of layers
        bstr += len(neighs_list).to_bytes(I_SIZE, byteorder=BYTE_ORDER)     # <N_HOODS>
        logger.debug(f"Neighborhoods len: {len(neighs_list)}")
        # iterate now in neighbors
        for layer, neighs_set in enumerate(neighs_list): 
            page_ids = [node.get_internal_page_id() for node in neighs_set]
            logger.debug(f"Processing L{layer} (neighs page ids: {page_ids}) ...")
            bstr += layer.to_bytes(I_SIZE, byteorder=BYTE_ORDER) +\
                     len(neighs_set).to_bytes(I_SIZE, byteorder=BYTE_ORDER) # <N_LAYER> <N_NEIGS>
            # get each internal page id of the neighs
            bstr += b''.join([page_id.to_bytes(I_SIZE, byteorder=BYTE_ORDER) for page_id in page_ids])
       
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

    @classmethod
    def load(cls, filename, distance_algorithm=None, db_manager=None):
        """Restores Apotheosis structure from permanent storage.
        
        Arguments:
        filename            -- filename to load
        distance_algorithm  -- distance algorithm to check in the file
        db_manager          -- db manager to retrieve other relevant data (we only keep page ids in permanent storage)
        """
        
        logger.info(f"Restoring Apotheosis structure from disk (filename \"{filename}\", distance algorithm {distance_algorithm}\") ...")
        newAPO = Apotheosis(filename=filename, distance_algorithm=distance_algorithm, db_manager=db_manager)
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

    def draw(self, filename: str, show_distance: bool=True, format="pdf", cluster: bool=False):
        """Creates a graph figure per level of the HNSW structure and saves it to a filename file.

        Arguments:
        filename        -- filename to create (with extension)
        show_distance   -- to show the distance metric in the edges (default is True)
        format          -- matplotlib plt.savefig(..., format=format) (default is "pdf")
        cluster         -- bool flag to draw also the structure in cluster mode (considering modules)
        """
        logger.info(f"Drawing to {filename} (with cluster? {cluster}) ...")
        self._HNSW.draw(filename, show_distance=show_distance, format=format, cluster=cluster)

    # to support ==, now the object is not unhasheable (cannot be stored in sets or dicts)
    def __eq__(self, other):
        """Returns True if this object and other are the same, False otherwise.

        Arguments:
        other   -- HNSW object to check
        """
        if type(self) != type(other):
            return False

        return self._HNSW == other._HNSW

# unit test
import common.utilities as util
from datalayer.node.hash_node import HashNode
from datalayer.node.winmodule_hash_node import WinModuleHashNode
from datalayer.hash_algorithm.tlsh_algorithm import TLSHHashAlgorithm
from datalayer.hash_algorithm.ssdeep_algorithm import SSDEEPHashAlgorithm
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
        exact_found, results = apo.knn_search(query=query_node, k=2, ef=4)
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
                    beer_factor=args.beer_factor,
                    distance_algorithm=TLSHHashAlgorithm)

    # Create the nodes based on TLSH Fuzzy Hashes
    hash1 = "T1BF81A292E336D1F68224D4A4C751A2B3BB353CA9C2103BA69FA4C7908761B50F22E301" #fake
    hash2 = "T12B81E2134758C0E3CA097B381202C62AC793B46686CD9E2E8F9190EC89C537B5E7AF4C" 
    hash3 = "T10381E956C26225F2DAD9D5C2C5C1A337FAF3708A25012B8A1EACDAC00B37D557E0E714"
    hash4 = "T1DF8174A9C2A506F9C6FFC292D6816333FEF1B845C419121A0F91CF5359B5B21FA3A304" 
    hash5 = "T1DF8174A9C2A506F9C6FFC292D6816333FEF1B845C419121A0F91CF5359B5B21FA3A305" #fake
    hash6 = "T1DF8174A9C2A506FC122292D644816333FEF1B845C419121A0F91CF5359B5B21FA3A305" #fake
    hash7 = "T10381E956C26225F2DAD9D097B381202C62AC793B37082B8A1EACDAC00B37D557E0E714" #fake

    node1 = WinModuleHashNode(hash1, TLSHHashAlgorithm)
    node2 = WinModuleHashNode(hash2, TLSHHashAlgorithm)
    node3 = WinModuleHashNode(hash3, TLSHHashAlgorithm)
    node4 = WinModuleHashNode(hash4, TLSHHashAlgorithm)
    node5 = WinModuleHashNode(hash5, TLSHHashAlgorithm)
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
    #print(f"Restoring Apotheosis at {date_time} ...")
    #myAPO = Apotheosis.load("myAPO_uncompressed"+date_time, TLSHHashAlgorithm)
    #myAPO = Apotheosis.load("myAPO"+date_time)

    # cluster test
    in_cluster = 10 # random nodes in the cluster
    alphabet = []
    for i in range(0, 10): # '0'..'9'
        alphabet.append(str(i + ord('0')))
    
    for i in range(0, 6): # 'A'..'F'
        alphabet.append(str(i + ord('0')))


    _nodes = []
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
            _nodes.add(node)
        except:
            continue

