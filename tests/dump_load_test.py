import statistics
import time
import logging
import sys
sys.setrecursionlimit(200000) # avoids pickle recursion error for large objects

import common.utilities as util
from datalayer.db_manager import DBManager

from apotheosis import Apotheosis
from datalayer.hash_algorithm.tlsh_algorithm import TLSHHashAlgorithm
from datalayer.hash_algorithm.ssdeep_algorithm import SSDEEPHashAlgorithm
from datalayer.node.winmodule_hash_node import WinModuleHashNode

from common.errors import NodeAlreadyExistsError

def main():

    parser = util.configure_argparse()
    parser.add_argument('-recall', '--search-recall', type=int, default=4, help="Search recall (default=4)")
    parser.add_argument('-dump', '--dump-file', type=str, help="Filename to dump Apotheosis data structure")
    parser.add_argument('--npages', type=int, default=1000, help="Number of pages to test (default=1000)")

    args = parser.parse_args()
    # set logging level
    util.configure_logging(args.loglevel.upper())

    algorithm = TLSHHashAlgorithm
    if args.distance_algorithm == "ssdeep":
        algorithm = SSDEEPHashAlgorithm

    page_hashes, all_pages, current_model = util.create_model(args.npages, 0,\
                                args.M, args.ef, args.Mmax, args.Mmax0,\
                                args.heuristic, not args.no_extend_candidates, not args.no_keep_pruned_conns,\
                                algorithm, args.beer_factor)
    print("=&=&=&=&=&=&=&=&=")
    
    filename = args.dump_file
    if filename:
        print(f"[*] Dumping to \"{filename}\" ...")
        current_model.dump(filename)
        print(f"[*] Loading from \"{filename}\" ...")
        db_manager = DBManager()
        model = Apotheosis.load(filename, distance_algorithm=algorithm, hash_node_class=WinModuleHashNode)
        equal = current_model == model
        if not equal:
            breakpoint()
        print("Loaded model == created model?", current_model == model)

# driver unit
if __name__ == "__main__":
    main()
