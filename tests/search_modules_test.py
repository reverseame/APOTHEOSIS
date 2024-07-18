import random
import sys
sys.setrecursionlimit(200000) # avoids pickle recursion error for large objects

import common.utilities as util
from datalayer.db_manager import DBManager

from apotheosis_winmodule import ApotheosisWinModule
from datalayer.hash_algorithm.tlsh_algorithm import TLSHHashAlgorithm
from datalayer.hash_algorithm.ssdeep_algorithm import SSDEEPHashAlgorithm
from datalayer.node.winmodule_hash_node import WinModuleHashNode

from common.errors import NodeAlreadyExistsError

def create_model(modules_of_interest, npages, M, ef, Mmax, Mmax0, heuristic, extend_candidates, keep_pruned_conns, distance_algorithm):
    dbManager = DBManager()
    print("[*] Getting DB pages ... ", end='')
    all_pages, modules = dbManager.get_winmodules(distance_algorithm, limit=npages, modules_of_interest=modules_of_interest)
    print("done!")
    print(f"[*] Building ApotheosisWinModule model ({M},{ef},{Mmax},{Mmax0}) ... ")
    current_model = ApotheosisWinModule(M=M, ef=ef, Mmax=Mmax, Mmax0=Mmax0, 
                        distance_algorithm=distance_algorithm)
    page_list = []
    for i in range(0, npages):
        try:
            if i >= len(all_pages):
                break
            current_model.insert(all_pages[i])
            page_list.append(all_pages[i].get_id())
        except NodeAlreadyExistsError: # we need to check the DB ... 
            print(f"Node \"{all_pages[i].get_id()}\" already exists!")
        pass
    print("[+] Model built!")

    #XXX if we close the session, later we loose the relantionship between pages and modules 
    #dbManager.close()
    return page_list, current_model

# driver unit
if __name__ == "__main__":
    parser = util.configure_argparse()
    parser.add_argument('-recall', '--search-recall', type=int, default=4, help="Search recall (default=4)")
    parser.add_argument('--npages', type=int, default=1000, help="Number of pages to test (default=1000)")
    parser.add_argument('--ncluster', type=int, default=100, help="Number of pages to cluster (default=100)")
    parser.add_argument('-mof','--modules-of-interest', nargs='+', help='List of DLLs to retrieve from database')
    parser.add_argument('-cluster',action='store_true',help='Draw clustering graph (per module)')
    args = parser.parse_args()
    # set logging level
    util.configure_logging(args.loglevel.upper())

    _algorithm = TLSHHashAlgorithm
    if args.distance_algorithm == "ssdeep":
        _algorithm = SSDEEPHashAlgorithm

    pages, current_model = create_model(args.modules_of_interest, args.npages, args.M, args.ef, args.Mmax, args.Mmax0,\
                                args.heuristic, not args.no_extend_candidates, not args.no_keep_pruned_conns,\
                                _algorithm)
    # create PDF file for each layer to facilite debugging purposes
    if args.draw:
        current_model.draw(f"_npages{args.npages}_ef{args.search_recall}.pdf", cluster=args.cluster)

    print("=&=&=&=&=&=&=&=&=")
    print(f"Selecting {args.ncluster} random hashes from retrieved pages ...")
    hash_cluster = set()
    for i in range(0, args.ncluster):
        hash_cluster.add(random.choice(pages)) # can have less than ncluster, but we don't care for testing
    
    print(f"Drawing {args.ncluster} random hashes to disk ...")
    current_model.draw_hashes_subset(hash_cluster, "cluster.pdf")
