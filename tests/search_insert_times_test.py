import statistics
import time
import logging
import sys
sys.setrecursionlimit(200000) # avoids pickle recursion error for large objects

from common import utilities as util
from datalayer.db_manager import DBManager

from apotheosis import Apotheosis
from datalayer.hash_algorithm.tlsh_algorithm import TLSHHashAlgorithm
from datalayer.hash_algorithm.ssdeep_algorithm import SSDEEPHashAlgorithm
from datalayer.node.winmodule_hash_node import WinModuleHashNode

from common.errors import NodeAlreadyExistsError

def pages_are_equal(idx, page1, page2):
    result, results = page1.is_equal(page2)
    return result, results

def create_model(all_pages, M, ef, Mmax, Mmax0, heuristic, extend_candidates, keep_pruned_conns, distance_algorithm, beer_factor):
    print(f"[*] Building Apotheosis model ({M},{ef},{Mmax},{Mmax0}) ... ")
    current_model = Apotheosis(M=M, ef=ef, Mmax=Mmax, Mmax0=Mmax0, 
                        heuristic=heuristic, extend_candidates=extend_candidates, 
                        keep_pruned_conns=keep_pruned_conns, distance_algorithm=distance_algorithm,
                        beer_factor=beer_factor)
    page_list = []
    insert_times = []
    for i in range(0, len(all_pages)):
        try:
            start = time.time_ns()
            current_model.insert(all_pages[i]) # can raise exception
            end = time.time_ns() # in nanoseconds
            insert_times.append((end - start)/(10**3)) # convert to ms
            page_list.append(all_pages[i].get_id())
        except NodeAlreadyExistsError: # it should never occur...
            # get module already in DB, and print it to compare with the other one
            exact, node = current_model.search_exact_match_only(all_pages[i].get_id())
            if not exact: # infeasible path. If you see this, something weird happened
                raise Exception # db was modified in the backend, don't worry ...
            
            # check they are _really_ the same
            existing_page = node.get_page()
            new_page = all_pages[i].get_page()
            equal, equal_test = pages_are_equal(i, existing_page, new_page)
            if equal:
                logging.warning(f"Node \"{node.get_id()}\" already exists (different page id, same hashes)!")
            else:
                logging.error(f"TLSH, SSDEEP, SDHASH: {equal_test[0]}, {equal_test[1]}, {equal_test[2]}")
                if equal_test[0] and new_page.hashTLSH != existing_page.hashTLSH:
                    print("SOMETHING WAS WRONG ... TLSH equal? {equal_test[0]},  but \"{new_page.hashTLSH}\" != {existing_page.hashTLSH}")
                if equal_test[1] and new_page.hashSSDEEP != existing_page.hashSSDEEP:
                    print("SOMETHING WAS WRONG ... SSDEEP equal? {equal_test[1]},  but \"{new_page.hashSSDEEP}\" != {existing_page.hashSSDEEP}")
                if equal_test[2] and new_page.hashSD != existing_page.hashSD:
                    print("SOMETHING WAS WRONG ... SDHASH equal? {equal_test[2]},  but \"{new_page.hashSD}\" != {existing_page.hashSD}")
                logging.error(f"Some hash collision occurred with: {existing_page} vs {new_page}") # gold mine is here, guys
                #print(f'Arg! this is really a collision? {existing_page} vs {new_page}!')    # may happen with weak hash functions
        pass
    avg_insert_times = statistics.mean(insert_times)
    print(f"[+] INSERT Elapsed time: {avg_insert_times}")
    print("[+] Model built!")

    #dbManager.close()
    return page_list, current_model

def search(pages_search, current_model, search_recall):
    precision = 0
    search_times = []
    for idx, hash_value in enumerate(pages_search):
        start = time.time_ns() # in nanoseconds
        exact, hashes = current_model.knn_search(pages_search[idx], 1, ef=search_recall)
        end = time.time_ns()
        search_times.append((end - start)/(10**3)) # convert to ms
        if exact:
            precision += 1
        else:
            print(f"Hash \"{hash_value}\" not found. Value returned: {hashes}")
    
    avg_search_times = statistics.mean(search_times)
    return avg_search_times, precision

def main():
    parser = util.configure_argparse()
    parser.add_argument('-recall', '--search-recall', type=int, default=4, help="Search recall (default=4)")
    parser.add_argument('-dump', '--dump-file', type=str, help="Filename to dump Apotheosis data structure")
    parser.add_argument('--npages', type=int, default=1000, help="Number of pages to test (default=1000)")
    parser.add_argument('--nsearch-pages', type=int, default=0, help='Number of pages to search from outside the model (using HNSW)') 
    args = parser.parse_args()
    # set logging level
    util.configure_logging(args.loglevel.upper())
    algorithm = TLSHHashAlgorithm
    if args.distance_algorithm == "ssdeep":
        algorithm = SSDEEPHashAlgorithm

    print("[*] Getting DB pages ... ")
    db_manager = DBManager()
    all_pages, _ = db_manager.get_winmodules(algorithm, args.npages + args.nsearch_pages) 

    pages_insert = all_pages[:args.npages]
    inserted_pages, current_model = create_model(pages_insert, args.M, args.ef, args.Mmax, args.Mmax0,\
                                args.heuristic, not args.no_extend_candidates, not args.no_keep_pruned_conns,\
                                algorithm, args.beer_factor)
    # create PDF file for each layer to facilite debugging purposes
    if args.draw:
        current_model.draw(f"_npages{args.npages}_ef{args.search_recall}.pdf")

    print("=&=&=&=&=&=&=&=&=")
    print(f"[*] Starting search recall test with recall={args.search_recall}, heuristic={args.heuristic} ... ")

    avg_search_times, precision = search(pages_insert, current_model, args.search_recall)
    print(f"[+] SEARCH exact: {avg_search_times}") 
    print(f"[+] Precision: {precision}/{len(inserted_pages)} " + "({:.2f}%) {}OK".format(precision*100/len(inserted_pages), "" if precision == len(inserted_pages) else "N"))
    
    if args.nsearch_pages:
        search_pages = all_pages[:args.nsearch_pages]
        avg_search_times, _ = search(search_pages, current_model, args.search_recall)
        print(f"[+] SEARCH AKNN: {avg_search_times}") 

    filename = args.dump_file
    if filename:
        print(f"[*] Dumping to \"{filename}\" ...")
        current_model.dump(filename)
        print(f"[*] Loading from \"{filename}\" ...")
        model = Apotheosis.load(filename, distance_algorithm=algorithm, db_manager=db_manager)
        equal = current_model == model
        if not equal:
            breakpoint()
        print("Loaded model == created model?", current_model == model)

# driver unit
if __name__ == "__main__":
    main()
