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
from datalayer.errors import NodeAlreadyExistsError

def pages_are_equal(idx, page1, page2):
    result = page1.is_equal(page2)
    return result
    #XXX FAULTY unsafe, str == becomes flawed
    #return page1.hashTLSH == page2.hashTLSH and page1.hashSSDEEP == page2.hashSSDEEP and page1.hashSD == page2.hashSD 

def create_model(npages, M, ef, Mmax, Mmax0, heuristic, extend_candidates, keep_pruned_conns, distance_algorithm, beer_factor):
    dbManager = DBManager()
    print("[*] Getting DB pages ... ", end='')
    all_pages, module_list = dbManager.get_winmodules(distance_algorithm, npages)
    print("done!")
    print(f"[*] Building Apotheosis model ({M},{ef},{Mmax},{Mmax0}) ... ")
    current_model = Apotheosis(M=M, ef=ef, Mmax=Mmax, Mmax0=Mmax0, 
                        distance_algorithm=distance_algorithm, beer_factor=beer_factor)
    page_list = []
    insert_times = []
    for i in range(0, npages):
        try:
            start = time.time()
            current_model.insert(all_pages[i])
            end = time.time()
            insert_times.append(end - start)
            page_list.append(all_pages[i].get_id())
        except NodeAlreadyExistsError: # it should never occur...
            # get module already in DB, and print it to compare with the other one
            exact, node = current_model.search_exact_match_only(all_pages[i].get_id())
            if not exact: # infeasible path. If you see this, something weird happened
                raise Exception # db was modified in the backend, don't worry ...
            
            # check they are _really_ the same
            existing_page = node.get_page()
            new_page = all_pages[i].get_page()
            if pages_are_equal(i, existing_page, new_page):
                logging.warning(f"Node \"{node.get_id()}\" already exists (different page id, same hashes)!")
            else:
                logging.error(f"Some hash collision occurred with: {existing_page} vs {new_page}") # gold mine is here, guys
                #print(f'Arg! this is really a collision? {existing_page} vs {new_page}!')    # may happen with weak hash functions
        pass
    avg_insert_times = statistics.mean(insert_times)
    print(f"[+] INSERT Elapsed time: {avg_insert_times}")
    print("[+] Model built!")

    #dbManager.close()
    return page_list, all_pages, current_model

# driver unit
if __name__ == "__main__":
    parser = util.configure_argparse()
    parser.add_argument('-recall', '--search-recall', type=int, default=4, help="Search recall (default=4)")
    parser.add_argument('--npages', type=int, default=1000, help="Number of pages to test (default=1000)")

    args = parser.parse_args()
    # set logging level
    util.configure_logging(args.loglevel.upper())

    _algorithm = TLSHHashAlgorithm
    if args.distance_algorithm == "ssdeep":
        _algorithm = SSDEEPHashAlgorithm

    page_hashes, all_pages, current_model = create_model(args.npages, args.M, args.ef, args.Mmax, args.Mmax0,\
                                args.heuristic, not args.no_extend_candidates, not args.no_keep_pruned_conns,\
                                _algorithm, args.beer_factor)
    # create PDF file for each layer to facilite debugging purposes
    if args.draw:
        current_model.draw(f"_npages{args.npages}_ef{args.search_recall}.pdf")

    print("=&=&=&=&=&=&=&=&=")
    print(f"[*] Starting search recall test with recall={args.search_recall}, heuristic={args.heuristic} ... ")
    precision = 0
    search_times = []
    for idx, hash_value in enumerate(page_hashes):
        start = time.time()
        exact, hashes = current_model.knn_search(all_pages[idx], 1, ef=args.search_recall)
        end = time.time()
        search_times.append(end - start)
        if exact:
            precision += 1
        else:
            logger.info(f"Hash \"{page}\" not found. Value returned: {hashes}")
    
    avg_search_times = statistics.mean(search_times)
    print(f"[+] SEARCH Elapsed time: {avg_search_times}")

    #current_model.dump("test")
    print(f"[+] Precision: {precision}/{len(page_hashes)} " + "({:.2f}%) {}OK".format(precision*100/len(page_hashes), "" if precision == len(page_hashes) else "N"))
