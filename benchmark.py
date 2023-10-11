from hnsw import HNSW
from node_number import NumberNode
from node_hash import HashNode
from tlsh_algorithm import TLSHHashAlgorithm
from ssdeep_algorithm import SSDEEPHashAlgorithm
from sdhash_algorithm import SDHashAlgorithm
from db_manager import DBManager
import time
import tlsh
import random
import numpy as np

# CONFIG

MODELS = [(128, 128, 256, 256), (128, 128, 256, 512)]# [(4, 8, 8, 16), (16,32,32,64), (32, 64, 64, 128), \
         # (64, 64, 128, 256), (128, 128, 256, 256), (128, 128, 256, 512)]

ALT_MODELS = [(4, 8, 8, 8), (4, 8, 8, 16), (4, 8, 8, 32), (4, 8, 8, 64), \
              (4, 8, 16, 8), (4, 8, 32, 8), (4, 8, 64, 8), \
              (4, 16, 8, 8), (4, 32, 8, 8), (4, 64, 8, 8), \
              (8, 8, 8, 8), (16, 8, 8, 8), (32, 8, 8, 8)]

PERCENTAGE_OF_PAGES = [0.03, 0.10, 0.25, 0.5, 0.75, 1]
MAX_SEARCH_PERCENTAGES_SCORE = 60 
N_HASHES_SEARCH = 100
KNN = 10
ALGORITHM = SSDEEPHashAlgorithm

INSERTION_TIMES_FILENAME = "times_insertion"
INSERTION_TIMES_INDIVIDUAL_FILENAME = "individual_times_insertion"
SEARCH_KNN_FILENAME = "search_knn"
SEARCH_PERCENTAGE_FILENAME = "search_percentage"
BRUTEFORCE_PERCENTAGE_FILENAME = "hnsw"

BENCHMARK_DIR = "benchmarks"
BENCHMARK_SUBDIR = ""
if ALGORITHM == TLSHHashAlgorithm:
    BENCHMARK_SUBDIR = "tlsh"
elif ALGORITHM == SSDEEPHashAlgorithm:
    BENCHMARK_SUBDIR = "ssdeep"
elif ALGORITHM == SDHashAlgorithm:
    BENCHMARK_SUBDIR = "sdhash"

def get_db_pages():
    dbManager = DBManager()
    print("Getting pages from DB...")
    if ALGORITHM == TLSHHashAlgorithm:
        return dbManager.get_all_pages()
    elif ALGORITHM == SSDEEPHashAlgorithm:
        return dbManager.get_all_pages_ssdeep()
    elif ALGORITHM == SDHashAlgorithm:
        return dbManager.get_all_pages_sdhash()
    

def search_node(model, hash, knn):
    init_time = time.time()
    nodes = model.knn_search(HashNode(hash, PERCENTAGE_OF_PAGES), knn, model.ef)
    return (time.time() - init_time, nodes[0])

def search_node_percentage(model, hash, percentage):
    init_time = time.time()
    nodes = model.percentage_search(HashNode(hash, PERCENTAGE_OF_PAGES), percentage)
    return (time.time() - init_time, list(nodes))
     
def search_nodes_percentage(model, n, hashes, percentage):    # Imprimir hashes también
    search_times = []
    search_precissions = []
    n_hashes = len(hashes)
    for i in range(0, n_hashes):
        time, nodes = search_node_percentage(model, hashes[i], percentage)
        search_times.append(time)
        precissions_per_hash = []
        for node in nodes:
            precissions_per_hash.append(ALGORITHM.compare(hashes[i], node.id))
        search_precissions.append(precissions_per_hash)
    write_file(SEARCH_KNN_FILENAME, time, nodes, model, n, percentage)


def write_file(file_prefix, search_times, search_precisions, model, n, percentage):
    with open(f'{BENCHMARK_DIR}/{BENCHMARK_SUBDIR}/{file_prefix}_{model.M}_{model.ef}_{model.Mmax}_{model.Mmax0}_{n}_{percentage}.txt', 'w') as f:
        for i in range(0, N_HASHES_SEARCH):
            f.write("[%s]: %s\n" % (search_times[i], str(search_precisions[i])))

def write_file_alt(file_prefix, search_times, search_precisions, model, n, percentage):
    with open(f'{BENCHMARK_DIR}/{BENCHMARK_SUBDIR}/{file_prefix}_{model.M}_{model.ef}_{model.Mmax}_{model.Mmax0}_{n}_{percentage}.txt', 'w') as f:
        for i in range(0, 84136):
            f.write("[%s]: %s\n" % (search_times[i], str(search_precisions[i])))

def write_insertion_times(add_times, model, n):
    with open(f'{BENCHMARK_DIR}/{BENCHMARK_SUBDIR}/{INSERTION_TIMES_FILENAME}_{model.M}_{model.ef}_{model.Mmax}_{model.Mmax0}_{n}.txt', 'w') as f:
        for item in add_times:
            f.write("%s\n" % item)
 
def perform_insertion_benchmark(model, list_pages, n_pages):
    add_times = []
    for i in range(0, n_pages):
        init_time = time.time()
        model.add_node(HashNode(list_pages[i], ALGORITHM))
        end_time = time.time()
        print(f"Elapsed time: {end_time - init_time}")
        add_times.append(end_time - init_time)
        print(f"\rAdding {i}/{n_pages} pages", end='', flush=True)
    write_insertion_times(add_times, model, n_pages)

def perform_search_percentage_benchmark(model, n_pages, hashes, percentage):
    search_times = []
    search_precissions = []
    n_hashes = len(hashes)
    for i in range(0, n_hashes):
        init_time = time.time()
        nodes = model.percentage_search(HashNode(hashes[i], ALGORITHM), percentage)
        elapsed_time = time.time() - init_time
        search_times.append(elapsed_time)
        precissions_per_hash = []
        for node in list(nodes):
            precissions_per_hash.append(ALGORITHM.compare(hashes[i], node.id))
        search_precissions.append(precissions_per_hash)
    write_file(SEARCH_PERCENTAGE_FILENAME, search_times, search_precissions, model, n_pages, percentage)

def perform_search_knn_benchmark(model, n_pages, hashes, knn):
    search_times = []
    search_precisions = []
    n_hashes = len(hashes)
    for i in range(0, n_hashes):
        init_time = time.time()
        nodes = model.knn_search(HashNode(hashes[i], ALGORITHM), knn, model.ef)
        elapsed_time = time.time() - init_time
        search_times.append(elapsed_time)
        precisions_per_hash = []
        for node in list(nodes):
            precisions_per_hash.append(ALGORITHM.compare(hashes[i], node.id))
        search_precisions.append(precisions_per_hash)
    write_file(SEARCH_KNN_FILENAME, search_times, search_precisions, model, n_pages, knn)

def perform_individual_parameters_benchmark(list_pages):
    n_pages = int(PERCENTAGE_OF_PAGES[0]*len(list_pages))
    hashes = random.sample(list_pages[n_pages:], 100)
    for model in ALT_MODELS:
        current_model = HNSW(*model)
        perform_insertion_benchmark(current_model, list_pages, n_pages)
        perform_search_knn_benchmark(current_model, n_pages, hashes, 1)

def perform_precision_bruteforce_benchmark(model, list_pages):
    n_pages = int(PERCENTAGE_OF_PAGES[0]*len(list_pages))
    similarities = []
    search_times = []
    for i in range(0, n_pages):
        init_time = time.time()
        nodes = model.percentage_search(HashNode(list_pages[i], ALGORITHM), 0.7)
        elapsed_time = time.time() - init_time
        precissions_per_hash = []
        for node in list(nodes):
            precissions_per_hash.append(ALGORITHM.compare(list_pages[i], node.id))
        similarities.append(precissions_per_hash)
        search_times.append(elapsed_time)

    print(len(similarities))
    print(len(search_times))
    write_file_alt(BRUTEFORCE_PERCENTAGE_FILENAME, search_times, similarities, model, n_pages, 0.7)
    #np.savetxt(f'{BENCHMARK_DIR}/{BENCHMARK_SUBDIR}/hnsw_{model.M}_{model.ef}_{model.Mmax}_{model.Mmax0}_.txt', similarities, fmt='%d')
        
def perform_benchmark():
    list_pages = get_db_pages()
    for percentage in PERCENTAGE_OF_PAGES:
        n_pages = int(percentage*len(list_pages))
        #hashes = random.sample(list_pages[n_pages:], 100)
        hashes = random.sample(list_pages, 100)
        for model in MODELS:
            print("Benchmarking model ({}, {}, {}, {}) with {} pages".format(*model, n_pages))
            current_model = HNSW(*model)
            perform_insertion_benchmark(current_model, list_pages, n_pages)
            perform_search_knn_benchmark(current_model, n_pages, hashes, 1)
            perform_search_knn_benchmark(current_model, n_pages, hashes, 10)
            perform_search_percentage_benchmark(current_model, n_pages, hashes, MAX_SEARCH_PERCENTAGES_SCORE)
            perform_precision_bruteforce_benchmark(current_model, list_pages)
    
    #perform_individual_parameters_benchmark(list_pages)


perform_benchmark()