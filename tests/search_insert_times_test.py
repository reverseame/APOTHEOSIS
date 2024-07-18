import statistics
import time
import logging
import sys
sys.setrecursionlimit(200000) # avoids pickle recursion error for large objects

from common import utilities as util
from datalayer.db_manager import DBManager

from apotheosis_winmodule import ApotheosisWinModule
from datalayer.hash_algorithm.tlsh_algorithm import TLSHHashAlgorithm
from datalayer.hash_algorithm.ssdeep_algorithm import SSDEEPHashAlgorithm
from datalayer.node.winmodule_hash_node import WinModuleHashNode

from common.errors import NodeAlreadyExistsError

def search(pages_search, current_model, search_recall):
    precision = 0
    search_times = []
    for idx in range(0, len(pages_search)):
        hash_value = pages_search[idx]
        start = time.time_ns() # in nanoseconds
        exact, node, hashes = current_model.knn_search(hashid=hash_value, k=1, ef=search_recall)
        end = time.time_ns()
        search_times.append((end - start)/(1e6)) # convert to ms
        if exact:
            precision += 1
        else:
            print(f"Hash \"{hash_value}\" not found. Value returned: {hashes}")
    
    avg_search_times = statistics.mean(search_times)
    return avg_search_times, precision

def main():
    parser = util.configure_argparse()
    parser.add_argument('-recall', '--search-recall', type=int, default=4, help="Search recall (default=4)")
    parser.add_argument('-dump', '--dump-file', type=str, help="Filename to dump ApotheosisWinModule data structure")
    parser.add_argument('-np', '--npages', type=int, default=1000, help="Number of pages to test (default=1000)")
    parser.add_argument('-ns', '--nsearch-pages', type=int, default=0, help='Number of pages to search from outside the model (using HNSW)') 
    args = parser.parse_args()
    # set logging level
    util.configure_logging(args.loglevel.upper())
    algorithm = TLSHHashAlgorithm
    if args.distance_algorithm == "ssdeep":
        algorithm = SSDEEPHashAlgorithm

    print("[*] Getting DB pages ... ")

    npages = args.npages
    inserted_pages, all_pages, current_model = util.create_model(npages, args.nsearch_pages,\
                                args.M, args.ef, args.Mmax, args.Mmax0,\
                                args.heuristic, not args.no_extend_candidates, not args.no_keep_pruned_conns,\
                                algorithm, args.beer_factor)
    no_inserted = npages - len(inserted_pages)
    # create PDF file for each layer to facilite debugging purposes
    if args.draw:
        current_model.draw(f"_npages{args.npages}_ef{args.search_recall}.pdf")

    print("=&=&=&=&=&=&=&=&=")
    print(f"[*] Starting search recall test with recall={args.search_recall}, heuristic={args.heuristic} ... ")

    avg_search_times, precision = search(inserted_pages, current_model, args.search_recall)
    print(f"[+] SEARCH EXACT: {avg_search_times} ms")
    precision = precision - no_inserted # remove repeated entries not really inserted, but tested on the search
    print(f"[+] Precision: {precision}/{len(inserted_pages) - no_inserted} " + "({:.2f}%) {}OK".format(precision*100/(len(inserted_pages) - no_inserted), "" if precision + no_inserted == len(inserted_pages) else "N"))
    
    if args.nsearch_pages:
        search_pages = all_pages[-args.nsearch_pages:]
        search_pages = [node.get_id() for node in search_pages]
        avg_search_times, _ = search(search_pages, current_model, args.search_recall)
        print(f"[+] SEARCH AKNN: {avg_search_times} ms") 

    filename = args.dump_file
    if filename:
        print(f"[*] Dumping to \"{filename}\" ...")
        current_model.dump(filename)
        print(f"[*] Loading from \"{filename}\" ...")
        model = ApotheosisWinModule.load(filename, distance_algorithm=algorithm, db_manager=db_manager)
        equal = current_model == model
        if not equal:
            breakpoint()
        print("Loaded model == created model?", current_model == model)

# driver unit
if __name__ == "__main__":
    main()
