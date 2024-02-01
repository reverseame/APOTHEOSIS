import logging
import argparse

from db_manager import DBManager

from hnsw import HNSW
from datalayer.hash_algorithm.tlsh_algorithm import TLSHHashAlgorithm
from datalayer.hash_algorithm.ssdeep_algorithm import SSDEEPHashAlgorithm
from datalayer.node.node_hash import HashNode
from datalayer.errors import NodeAlreadyExistsError

def create_model(npages, M, ef, Mmax, Mmax0, heuristic, extend_candidates, keep_pruned_conns, distance_algorithm):
    dbManager = DBManager()
    print("Getting DB pages ... ", end='')
    all_node_pages = dbManager.get_winmodules(distance_algorithm, npages)
    print("done!")
    print(f"Building HNSW model ({M},{ef},{Mmax},{Mmax0}) ... ", end='')
    current_model = HNSW(M=M, ef=ef, Mmax=Mmax, Mmax0=Mmax0, 
                        distance_algorithm=distance_algorithm)
    print("done!")
    for i in range(0, npages):
        try:
            current_model.add_node(HashNode(all_node_pages[i].get_id(), distance_algorithm))
        except NodeAlreadyExistsError: # it should never occur...
            print(f"Node {all_node_pages[i].get_id()} already exists!")
        pass

    dbManager.close()
    return all_node_pages, current_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-recall', '--search-recall', type=int, default=4, help="Search recall (default=4)")
    parser.add_argument('--npages', type=int, default=1000, help="Number of pages to test (default=1000)")
    parser.add_argument('--M', type=int, default=4, help="Number of established connections of each node (default=4)")
    parser.add_argument('--ef', type=int, default=4, help="Exploration factor (determines the search recall, default=4)")
    parser.add_argument('--Mmax', type=int, default=8, help="Max links allowed per node at any layer, but layer 0 (default=8)")
    parser.add_argument('--Mmax0', type=int, default=16, help="Max links allowed per node at layer 0 (default=16)")
    parser.add_argument('--heuristic', help="Create a HNSW structure using a heuristic to select neighbors rather than a simple selection algorithm (disabled by default)", action='store_true')
    parser.add_argument('--no-extend-candidates', help="Neighbor heuristic selection extendCandidates parameter (enabled by default)", action='store_true')
    parser.add_argument('--no-keep-pruned-conns', help="Neighbor heuristic selection keepPrunedConns parameter (enabled by default)", action='store_true')
    parser.add_argument('-algorithm', '--distance-algorithm', choices=["tlsh", "ssdeep"], default='tlsh', help="Distance algorithm to be used in the HNSW structure (default=tlsh)")
    parser.add_argument('-log', '--loglevel', choices=["debug", "info", "warning", "error", "critical"], default='warning', help="Provide logging level (default=warning)")

    args = parser.parse_args()
    logger = logging.getLogger("hnsw")
    logging.basicConfig() # enables logging at module hnsw
    logger.setLevel(args.loglevel.upper())

    _algorithm = TLSHHashAlgorithm
    if args.distance_algorithm == "ssdeep":
        _algorithm = SSDEEPHashAlgorithm

    pages, current_model = create_model(args.npages, args.M, args.ef, args.Mmax, args.Mmax0,\
                                args.heuristic, not args.no_extend_candidates, not args.no_keep_pruned_conns,\
                                _algorithm)

    print("=&=&=&=&=&=&=&=&=")
    print(f"Starting search recall test with recall={args.search_recall}, heuristic={args.heuristic} ... ")
    precision = 0
    for page in pages:
        hashes = current_model.knn_search(HashNode(page.get_id(), _algorithm), 1, ef=args.search_recall)
        #print(f"Looking for {page.id} but found {hashes[0]}")
        #for hash in hashes:
        #    print(f" -> Sim: {HashNode(page.id, TLSHHashAlgorithm).calculate_similarity(HashNode(hash.id, TLSHHashAlgorithm))}")
        if len(hashes.keys()) == 1:
            _, _value = hashes.popitem()
            if page.get_id() == _value[0].get_id():
                precision += 1

    print(f"Precision: {precision}/{args.npages}")
