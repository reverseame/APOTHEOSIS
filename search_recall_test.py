import sys
import logging
import argparse

from db_manager import DBManager

from apotheosis import Apotheosis
from datalayer.hash_algorithm.tlsh_algorithm import TLSHHashAlgorithm
from datalayer.hash_algorithm.ssdeep_algorithm import SSDEEPHashAlgorithm
from datalayer.node.node_hash import HashNode
from datalayer.errors import NodeAlreadyExistsError

def create_model(npages, M, ef, Mmax, Mmax0, heuristic, extend_candidates, keep_pruned_conns, distance_algorithm):
    dbManager = DBManager()
    print("[*] Getting DB pages ... ", end='')
    all_node_pages = dbManager.get_winmodules(distance_algorithm, npages)
    print("done!")
    print(f"[*] Building Apotheosis model ({M},{ef},{Mmax},{Mmax0}) ... ")
    current_model = Apotheosis(M=M, ef=ef, Mmax=Mmax, Mmax0=Mmax0, 
                        distance_algorithm=distance_algorithm)
    _page_list = []
    for i in range(0, npages):
        try:
            current_model.add_node(HashNode(all_node_pages[i].get_id(), distance_algorithm))
            _page_list.append(all_node_pages[i].get_id())
        except NodeAlreadyExistsError: # it should never occur...
            print(f"Node \"{all_node_pages[i].get_id()}\" already exists!")
        pass
    print("[+] Model built!")

    dbManager.close()
    return _page_list, current_model

# https://stackoverflow.com/questions/54366106/configure-formatting-for-root-logger
def configure_logging(loglevel, logger=None):
    """
    Configures a simple console logger with the given level.
    A usecase is to change the formatting of the default handler of the root logger
    """
    #formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter('%(levelname)s:%(name)s: %(message)s')
    logger = logger or logging.getLogger()  # either the given logger or the root logger
    logger.setLevel(loglevel)
    # If the logger has handlers, we configure the first one. Otherwise we add a handler and configure it
    if logger.handlers:
        console = logger.handlers[0]  # we assume the first handler is the one we want to configure
    else:
        console = logging.StreamHandler()
        logger.addHandler(console)
    console.setFormatter(formatter)
    console.setLevel(loglevel)

# driver unit
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-recall', '--search-recall', type=int, default=4, help="Search recall (default=4)")
    parser.add_argument('--npages', type=int, default=1000, help="Number of pages to test (default=1000)")
    parser.add_argument('--M', type=int, default=4, help="Number of established connections of each node (default=4)")
    parser.add_argument('--ef', type=int, default=4, help="Exploration factor (determines the search recall, default=4)")
    parser.add_argument('--Mmax', type=int, default=8, help="Max links allowed per node at any layer, but layer 0 (default=8)")
    parser.add_argument('--Mmax0', type=int, default=16, help="Max links allowed per node at layer 0 (default=16)")
    parser.add_argument('--heuristic', help="Create an Apotheosis with the HNSW structure using a heuristic to select neighbors rather than a simple selection algorithm (disabled by default)", action='store_true')
    parser.add_argument('--no-extend-candidates', help="Neighbor heuristic selection extendCandidates parameter (enabled by default)", action='store_true')
    parser.add_argument('--no-keep-pruned-conns', help="Neighbor heuristic selection keepPrunedConns parameter (enabled by default)", action='store_true')
    parser.add_argument('-algorithm', '--distance-algorithm', choices=["tlsh", "ssdeep"], default='tlsh', help="Distance algorithm to be used in the Apotheosis structure (default=tlsh)")
    parser.add_argument('-draw', help="Draws the HNSW structure associated to Apotheosis to a file", action='store_true')
    parser.add_argument('-log', '--loglevel', choices=["debug", "info", "warning", "error", "critical"], default='warning', help="Provide logging level (default=warning)")

    args = parser.parse_args()
    # set logging level
    configure_logging(args.loglevel.upper())

    _algorithm = TLSHHashAlgorithm
    if args.distance_algorithm == "ssdeep":
        _algorithm = SSDEEPHashAlgorithm

    pages, current_model = create_model(args.npages, args.M, args.ef, args.Mmax, args.Mmax0,\
                                args.heuristic, not args.no_extend_candidates, not args.no_keep_pruned_conns,\
                                _algorithm)
    # create PDF file for each layer to facilite debugging purposes
    if args.draw:
        current_model.draw(f"_npages{args.npages}_ef{args.search_recall}.pdf")

    print("=&=&=&=&=&=&=&=&=")
    print(f"[*] Starting search recall test with recall={args.search_recall}, heuristic={args.heuristic} ... ")
    precision = 0
    for page in pages:
        found, hashes = current_model.knn_search(HashNode(page, _algorithm), 1, ef=args.search_recall)
        if found:
            precision += 1
        else:
            logger.info(f"Hash \"{page}\" not found. Value returned: {hashes}")

    print(f"[+] Precision: {precision}/{len(pages)} " + "({:.2f}%)".format(precision*100/len(pages)))
