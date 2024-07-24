from apotheosis import Apotheosis
from datalayer.hash_algorithm.tlsh_algorithm import TLSHHashAlgorithm
from datalayer.hash_algorithm.ssdeep_algorithm import SSDEEPHashAlgorithm
from common.errors import NodeAlreadyExistsError
from datalayer.db_manager import DBManager
from datalayer.node.hash_node import HashNode
import time
import random
import numpy as np
import logging

import sys

# Config
PERCENTAGE_OF_PAGES = [0.03, 0.10, 0.25, 0.5, 0.75, 1]
THRESHOLD_SEARCH = 60
THRESHOLD_SEARCH_NHOPS = 4
HNSW_CONFIGS = [(4, 8, 8, 16), (16,32,32,64), (32, 64, 64, 128), \
            (64, 64, 128, 256), (128, 128, 256, 256), (128, 128, 256, 512)]

INSERTION_TIMES_FILENAME = "times_insertion"
SEARCH_KNN_FILENAME = "search_knn"
SEARCH_THRESHOLD_FILENAME = "search_thershold"

BENCHMARK_DIR = "benchmarks"
BENCHMARK_SUBDIR = ""

def perform_insertion_benchmark(model, hnsw_config, list_pages, n_pages, distance_algorithm):
    add_times = []
    for i in range(0, n_pages):
        init_time = time.time_ns()
        try:
            model.insert(list_pages[i])
            elapsed_time = time.time_ns() - init_time
            add_times.append(elapsed_time)
            if len(add_times) > 1000:
                avg_value = sum(add_times[-1000:]) / 1000
                print(f"\rAdding {i}/{n_pages} pages {list_pages[i].get_id()} | "
                      f"ETA: {ns_to_hh_mm(avg_value, n_pages - i)} - "
                      f"Elapsed time: {ns_to_hh_mm(sum(add_times), 1)}", end='', flush=True)

        except NodeAlreadyExistsError:
            pass

    np.savetxt(f'{BENCHMARK_DIR}/{distance_algorithm}/{INSERTION_TIMES_FILENAME}_'
              f'{hnsw_config[0]}_{hnsw_config[1]}_{hnsw_config[2]}_{hnsw_config[3]}_'
              f'{n_pages}.txt', add_times, fmt='%d')
    
def perform_search_knn_benchmark(model, hnsw_config, list_pages, knn, distance_algorithm):
    search_times = []
    n_hashes = len(list_pages)
    for i in range(0, n_hashes):
        init_time = time.time()
        model.knn_search(list_pages[i], knn)
        elapsed_time = time.time() - init_time
        search_times.append(elapsed_time)
    np.savetxt(f'{BENCHMARK_DIR}/{distance_algorithm}/{SEARCH_KNN_FILENAME}_'
               f'{hnsw_config[0]}_{hnsw_config[1]}_{hnsw_config[2]}_{hnsw_config[3]}_'
                f'{n_hashes}.txt', search_times, fmt='%d')

def perform_search_threshold_benchmark(model, hnsw_config, list_pages, percentage, distance_algorithm):
    search_times = []
    n_hashes = len(list_pages)
    for i in range(0, n_hashes):
        init_time = time.time()
        model.threshold_search(list_pages[i], percentage, hnsw_config[3])
        elapsed_time = time.time() - init_time
        search_times.append(elapsed_time)

    np.savetxt(f'{BENCHMARK_DIR}/{distance_algorithm}/{SEARCH_THRESHOLD_FILENAME}_'
              f'{hnsw_config[0]}_{hnsw_config[1]}_{hnsw_config[2]}_{hnsw_config[3]}_'
              f'{len(list_pages)}.txt', search_times, fmt='%d')

def ns_to_hh_mm(ns, n_pages):
    seconds = (ns * n_pages) / 1e9
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return "%02d:%02d" % (hours, minutes)

''' #TODO: Discuss
def perform_precision_vs_bruteforce_benchmark(model, list_pages):
    n_pages = int(PERCENTAGE_OF_PAGES[0]*len(list_pages))
    precisions_per_hash = []
    similarity_list = perform_bruteforce_benchmark(list_pages)
    for i in range(0, n_pages):
        precision = 0
        nodes = model.knn_search(HashNode(list_pages[i].get_id(), ALGORITHM), 10, model.get_ef())
        for hash in similarity_list[i]:
            for _, node_id in nodes.items():
                if hash == node_id:
                    precision += 1
            precision = 0
            precisions_per_hash.append(precision)

    np.savetxt(f'{BENCHMARK_DIR}/{BENCHMARK_SUBDIR}/vs_precision.txt', precisions_per_hash, fmt='%d')
'''

def perform_benchmark(percentage, all_node_pages, hnsw_config, heuristic, distance_algorithm):
    try:
        n_pages = int(percentage * len(all_node_pages))
        print("Benchmarking model ({}, {}, {}, {}) with {} pages".format(*hnsw_config, n_pages))
        current_model = Apotheosis(M=hnsw_config[0], ef=hnsw_config[1],\
                                  Mmax=hnsw_config[2], Mmax0=hnsw_config[3],\
                                  heuristic=heuristic, extend_candidates=False, \
                                  keep_pruned_conns=False,\
                                  distance_algorithm=_get_algorithm_instance(distance_algorithm))
        perform_insertion_benchmark(current_model, hnsw_config, all_node_pages, n_pages, distance_algorithm)
        #perform_search_knn_benchmark(current_model, hnsw_config, all_node_pages, 1, distance_algorithm)
        #perform_search_knn_benchmark(current_model, hnsw_config, all_node_pages, 10, distance_algorithm)
        #perform_search_threshold_benchmark(current_model, n_pages, pages_to_search, MAX_SEARCH_PERCENTAGES_SCORE)
        print("Now dumping...")
        sys.setrecursionlimit(200000)
        current_model.dump(f'{BENCHMARK_DIR}/{distance_algorithm}/'
                           f'bench_model_{hnsw_config[0]}_{hnsw_config[1]}'
                           f'_{hnsw_config[2]}_{hnsw_config[3]}')
    except Exception as e:
        print(e)

def _get_db_pages(algorithm):
    dbManager = DBManager()
    print("Getting DB pages...")
    algorithm = _get_algorithm_instance(algorithm)
    return dbManager.get_winmodules(algorithm)

def _get_algorithm_instance(algorithm):
    return TLSHHashAlgorithm if distance_algorithm == "tlsh"\
                         else SSDEEPHashAlgorithm


import argparse
import multiprocessing
import concurrent
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apotheosis benchmark")
    parser.add_argument('--distance-algorithm', '-da', required=True,
                        choices=['tlsh', 'ssdeep'],
                        help='Specify the hash algorithm (tlsh or ssdeep)')
    parser.add_argument('--heuristic', '-ha', action='store_true',
                         help='HNSW with heuristic', default=False)
    parser.add_argument('--loglevel', '-ll', default='INFO', 
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Specify the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)')
    args = parser.parse_args()

    # Set up logging configuration
    logging.basicConfig(level=args.loglevel.upper(), format='%(levelname)s: %(message)s')

    distance_algorithm = args.distance_algorithm
    heuristic = args.heuristic
    
    all_pages, _ = _get_db_pages(distance_algorithm)
    #shared_list_pages = manager.list(all_pages)
    perform_benchmark(1, all_pages, (14,4,14,28), False, "tlsh")
    '''
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
            futures = {}
            for percentage in PERCENTAGE_OF_PAGES:
                for hnsw_config in HNSW_CONFIGS:
                    print("Executing benchmark with {}%\ of pages and config {}".format(percentage, hnsw_config))
                    future = executor.submit(perform_benchmark, percentage,
                                             shared_list_pages, hnsw_config, heuristic, 
                                             distance_algorithm)
                    futures[future] = (hnsw_config, percentage)
            print("Waiting for all tasks to complete...")
            concurrent.futures.wait(futures)
            '''
