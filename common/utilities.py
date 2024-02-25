# common utilities used by more than one script in the project

import sys
import logging
import argparse

from datalayer.db_manager import DBManager
def create_model(npages, nsearch_pages,\
                M, ef, Mmax, Mmax0, heuristic, extend_candidates, keep_pruned_conns,\
                distance_algorithm, beer_factor):

    from apotheosis import Apotheosis # avoid circular deps
    print(f"[*] Building Apotheosis model ({M},{ef},{Mmax},{Mmax0}) from DB ... ")
    current_model = Apotheosis(M=M, ef=ef, Mmax=Mmax, Mmax0=Mmax0,
                        heuristic=heuristic, extend_candidates=extend_candidates,
                        keep_pruned_conns=keep_pruned_conns, distance_algorithm=distance_algorithm,
                        beer_factor=beer_factor)

    return load_DB_in_model(npages=npages, nsearch_pages=nsearch_pages, algorithm=distance_algorithm, current_model=current_model)

import statistics
import time
import datetime
from common.errors import NodeAlreadyExistsError
def load_DB_in_model(npages=None, nsearch_pages=None, algorithm=None, current_model=None):
    
    db_manager = DBManager()

    print(f"[*] Getting modules from DB (with {algorithm.__name__}) ...")
    start = time.time_ns()
    all_pages, _ = db_manager.get_winmodules(algorithm, npages)
    end = time.time_ns() # in nanoseconds
    db_time = (end - start)/1e6 # ms
    print(f"[*] {len(all_pages)} pages recovered from DB in {db_time} ms.")

    page_list = []
    insert_times = []
    for i in range(0, len(all_pages)):
        if i % 1e6 == 0:
            print(f"{i/1e6}e6 pages already inserted ({datetime.datetime.now()}) ...")

        try:
            start = time.time_ns()
            current_model.insert(all_pages[i]) # can raise exception
            end = time.time_ns() # in nanoseconds
            insert_times.append((end - start)/(1e6)) # convert to ms
            page_list.append(all_pages[i].get_id())
        except NodeAlreadyExistsError: # it should never occur...
            # get module already in DB, and print it to compare with the other one
            exact, node = current_model.search_exact_match_only(all_pages[i].get_id())
            if not exact: # infeasible path. If you see this, something weird happened
                raise Exception # db was modified in the backend, don't worry ...

            # check they are _really_ the same
            existing_page = node.get_page()
            new_page = all_pages[i].get_page()
            equal, equal_test = pages_are_equal(existing_page, new_page)
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
    
    print(f"All pages ({len(all_pages)}) inserted ({datetime.datetime.now()}) ...")
    avg_insert_times = statistics.mean(insert_times)
    print(f"[+] INSERT Elapsed time: {avg_insert_times} ms")
    print("[+] Model built!")

    #dbManager.close()
    return page_list, all_pages, current_model

def pages_are_equal(page1, page2):
    result, results = page1.is_equal(page2)
    return result, results

# https://stackoverflow.com/questions/54366106/configure-formatting-for-root-logger
def configure_logging(loglevel, logger=None):
    """
    Configures a simple console logger with the given level.
    A usecase is to change the formatting of the default handler of the root logger
    
    Arguments:
    loglevel    -- log level to set
    logger      -- specific logger to configure. If None, it will configure root logger
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

    logging.basicConfig(stream=sys.stderr) # log everything to stderr by default
    console.setFormatter(formatter)
    console.setLevel(loglevel)

def print_results(results: dict, show_keys=False):
    """Print a dictionary of (similarity score, hash) in the output.

    Arguments:
    results     -- dictionary to print
    show_keys   -- bool to print each key along with each hash of the dict
    """

    # iterate now in the results. If we sort the keys, we can get them ordered by similarity score
    keys = sorted(results.keys())

    idx = 1
    for key in keys:
        for node in results[key]:
            _str = f"Node ID {idx}: \"{node.get_id()}\""
            if show_keys:
                _str += f" (score: {key})"
            print(_str)
            idx += 1

def configure_argparse() -> argparse.ArgumentParser:
    """Configures argparse to receive HNSW parameters + loglevel."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--M', type=int, default=4, help="Number of established connections of each node (default=4)")
    parser.add_argument('--ef', type=int, default=4, help="Exploration factor (determines the search recall, default=4)")
    parser.add_argument('--Mmax', type=int, default=8, help="Max links allowed per node at any layer, but layer 0 (default=8)")
    parser.add_argument('--Mmax0', type=int, default=16, help="Max links allowed per node at layer 0 (default=16)")
    parser.add_argument('--heuristic', help="Create the underlying HNSW structure using a heuristic to select neighbors rather than a simple selection algorithm (disabled by default)", action='store_true')
    parser.add_argument('--no-extend-candidates', help="Neighbor heuristic selection extendCandidates parameter (enabled by default)", action='store_true')
    parser.add_argument('--no-keep-pruned-conns', help="Neighbor heuristic selection keepPrunedConns parameter (enabled by default)", action='store_true')
    parser.add_argument('-algorithm', '--distance-algorithm', choices=["tlsh", "ssdeep"], default='tlsh', help="Distance algorithm to be used in the underlying HNSW structure (default=tlsh)")
    parser.add_argument("-bf", "--beer-factor", type=float,default=0, help="Factor for random walks (value should be in [0, 1), default 0)")
    parser.add_argument('--draw', help="Draws the underlying HNSW structure to file (disabled by default)", action='store_true')
    # get log level from command line
    parser.add_argument('-log', '--loglevel', choices=["debug", "info", "warning", "error", "critical"], default='warning', help="Provide logging level (default=warning)")

    return parser
