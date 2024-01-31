from hnsw import HNSW
from datalayer.node.node_number import NumberNode
from datalayer.node.node_hash import HashNode
from datalayer.hash_algorithm.tlsh_algorithm import TLSHHashAlgorithm
from datalayer.hash_algorithm.ssdeep_algorithm import SSDEEPHashAlgorithm
from datalayer.errors import NodeAlreadyExistsError
from db_manager import DBManager
import time
import tlsh
import random
import numpy as np
import sys
#import page
import psutil
import multiprocessing
import concurrent.futures
import logging

# CONFIG

class PrintToFile:
    def __init__(self, filename):
        self.original_stdout = sys.stdout
        self.log_file = open(filename, 'w')

    def write(self, text):
        self.original_stdout.write(text)
        self.log_file.write(text)
        self.log_file.flush()

    def close(self):
        sys.stdout = self.original_stdout
        self.log_file.close()


PERCENTAGE_OF_PAGES = [0.03, 0.10, 0.25, 0.5, 0.75, 1]
MAX_SEARCH_PERCENTAGES_SCORE = 60 
N_HASHES_SEARCH = 100
KNN = 10
ALGORITHM = TLSHHashAlgorithm
MODELS = [(4, 8, 8, 16, ALGORITHM), (16,32,32,64, ALGORITHM), (32, 64, 64, 128, ALGORITHM), \
          (64, 64, 128, 256, ALGORITHM), (128, 128, 256, 256, ALGORITHM), (128, 128, 256, 512, ALGORITHM)]
ALT_MODELS = [(4, 8, 8, 8), (4, 8, 8, 16), (4, 8, 8, 32), (4, 8, 8, 64), \
              (4, 8, 16, 8), (4, 8, 32, 8), (4, 8, 64, 8), \
              (4, 16, 8, 8), (4, 32, 8, 8), (4, 64, 8, 8), \
              (8, 8, 8, 8), (16, 8, 8, 8), (32, 8, 8, 8)]

INSERTION_TIMES_FILENAME = "times_insertion"
MEMORY_USAGE_FILENAME = "times_memory"
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

def search_node_percentage(model, hash, percentage):
    init_time = time.time()
    nodes = model.threshold_search(HashNode(hash, PERCENTAGE_OF_PAGES), percentage, model.get_M())
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
    with open(f'{BENCHMARK_DIR}/{BENCHMARK_SUBDIR}/{file_prefix}_{model.get_M()}_{model.get_ef()}_{model.get_Mmax()}_{model.get_Mmax0()}_{n}_{percentage}.txt', 'w') as f:
        for i in range(0, N_HASHES_SEARCH):
            f.write("[%s]: %s\n" % (search_times[i], str(search_precisions[i])))

def write_file_alt(file_prefix, search_times, search_precisions, model, n, percentage):
    with open(f'{BENCHMARK_DIR}/{BENCHMARK_SUBDIR}/{file_prefix}_{model.get_M()}_{model.get_ef()}_{model.get_Mmax()}_{model.get_Mmax0()}_{n}_{percentage}.txt', 'w') as f:
        for i in range(0, len(search_times)):
            f.write("[%s]: %s\n" % (search_times[i], str(search_precisions[i])))

def write_insertion_times(add_times, model, n):
    with open(f'{BENCHMARK_DIR}/{BENCHMARK_SUBDIR}/{INSERTION_TIMES_FILENAME}_{model.get_M()}_{model.get_ef()}_{model.get_Mmax()}_{model.get_Mmax0()}_{n}.txt', 'w') as f:
        for item in add_times:
            f.write("%s\n" % item)

def write_insertion_times(add_times, model, n):
    with open(f'{BENCHMARK_DIR}/{BENCHMARK_SUBDIR}/{INSERTION_TIMES_FILENAME}_{model.get_M()}_{model.get_ef()}_{model.get_Mmax()}_{model.get_Mmax0()}_{n}.txt', 'w') as f:
        for item in add_times:
            f.write("%s\n" % item)

def write_memory_times(add_times, model, n):
    with open(f'{BENCHMARK_DIR}/{BENCHMARK_SUBDIR}/{MEMORY_USAGE_FILENAME}_{model.get_M()}_{model.get_ef()}_{model.get_Mmax()}_{model.get_Mmax0()}_{n}.txt', 'w') as f:
        for item in add_times:
            f.write("%s\n" % item)


def monitor_memory_usage(memory_queue, stop_event):
    process = psutil.Process()  # Get the current process

    while not stop_event.is_set():
        try:
            memory_queue.put(process.memory_info().rss / (1024 * 1024))
            time.sleep(1)
        except KeyboardInterrupt:
            break  # Exit the loop if KeyboardInterrupt is received
        except Exception as e:
            print(f"Error in monitor_memory_usage: {e}")



def perform_insertion_benchmark(model, list_pages, n_pages):
    add_times = []
    memory_queue = multiprocessing.Queue()
    stop_event = multiprocessing.Event()
    monitor_process = multiprocessing.Process(target=monitor_memory_usage, args=(memory_queue, stop_event))
    #monitor_process.start()

    try:
        for i in range(0, n_pages):
            init_time = time.time_ns()
            try:
                model.add_node(list_pages[i])
                elapsed_time = time.time_ns() - init_time
                add_times.append(elapsed_time)
                if len(add_times) > 1000:
                    avg_value = sum(add_times[-1000:]) / 1000
                    print(f"\rAdding {i}/{n_pages} pages {list_pages[i].get_id()} | ETA: {ns_to_hh_mm(avg_value, n_pages - i)} - Elapsed time: {ns_to_hh_mm(sum(add_times), 1)}", end='', flush=True)
            except NodeAlreadyExistsError:
                pass
                
    except KeyboardInterrupt:
        pass

    finally:
        #stop_event.set()
        #monitor_process.join()

        # Retrieve memory data from the queue
        memory_data = []
        while not memory_queue.empty():
            memory_data.append(memory_queue.get())

        write_insertion_times(add_times, model, n_pages)
        write_memory_times(memory_data, model, n_pages)

def ns_to_hh_mm(ns, n_pages):
    seconds = (ns * n_pages) / 1e9
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return "%02d:%02d" % (hours, minutes)

def perform_search_knn_benchmark(model, n_pages, list_pages, knn):
    search_times = []
    search_precisions = []
    n_hashes = len(list_pages)
    for i in range(0, n_hashes):
        init_time = time.time()
        nodes = model.knn_search(HashNode(list_pages[i].get_id(), ALGORITHM), knn, model.get_ef())
        elapsed_time = time.time() - init_time
        search_times.append(elapsed_time)
        precisions_per_hash = []
        for distance, _ in nodes.items():
            precisions_per_hash.append(distance)
        search_precisions.append(precisions_per_hash)
    write_file(SEARCH_KNN_FILENAME, search_times, search_precisions, model, n_pages, knn)

def perform_individual_parameters_benchmark(list_pages):
    n_pages = int(PERCENTAGE_OF_PAGES[0]*len(list_pages))
    hashes = random.sample(list_pages[n_pages:], 100)
    for model in ALT_MODELS:
        current_model = HNSW(*model)
        perform_insertion_benchmark(current_model, list_pages, n_pages)
        perform_search_knn_benchmark(current_model, n_pages, hashes, 1)

def perform_bruteforce_benchmark(list_pages):
    n_pages = int(PERCENTAGE_OF_PAGES[0]*len(list_pages))
    similarity_list = []
    for i in range(0, n_pages):
        similar_hashes = []
        for j in range(0, n_pages):
            distance = ALGORITHM.compare(list_pages[i].get_id(), list_pages[j].id)
            similar_hashes.append((distance, list_pages[j].id))
            similar_hashes.sort()
            similar_hashes = similar_hashes[:10]
        similarity_list.append(similar_hashes)

    return similarity_list
        

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


def perform_search_percentage_benchmark(model, n_pages, list_pages, percentage):
    search_times = []
    search_precissions = []
    n_hashes = len(list_pages)
    for i in range(0, n_hashes):
        init_time = time.time()
        nodes = model.threshold_search(HashNode(list_pages[i].get_id(), ALGORITHM), MAX_SEARCH_PERCENTAGES_SCORE, model.get_M())
        elapsed_time = time.time() - init_time
        search_times.append(elapsed_time)
        precissions_per_hash = []
        for distance, _ in nodes.items():
            precissions_per_hash.append(distance)
        search_precissions.append(precissions_per_hash)
    write_file(SEARCH_PERCENTAGE_FILENAME, search_times, search_precissions, model, n_pages, percentage)
        
import sys
from multiprocessing import shared_memory
from datalayer.node.node_winmodule import WinModuleHashNode
import mmap
import os

def perform_benchmark(percentage, model):
    try:
        all_node_pages = get_db_pages()
        n_pages = int(percentage * len(all_node_pages))
        hashes = random.sample(all_node_pages[n_pages:], 100)
        sys.setrecursionlimit(200000)
        print("Benchmarking model ({}, {}, {}, {}) with {} pages".format(*model, n_pages))
        current_model = HNSW(*model)
        perform_insertion_benchmark(current_model, all_node_pages, n_pages)
        perform_search_knn_benchmark(current_model, n_pages, hashes, 1)
        perform_search_knn_benchmark(current_model, n_pages, hashes, 10)
        perform_search_percentage_benchmark(current_model, n_pages, hashes, MAX_SEARCH_PERCENTAGES_SCORE)
        current_model.dump(f"{model.get_M()}_{model.get_ef()}_{model.get_Mmax()}_{model.get_Mmax0()}_{percentage}.hnsw")
    except Exception as e:
        print(f"Exception in worker {os.getpid()}: {e}")

    #perform_precision_bruteforce_benchmark(current_model, all_node_pages)

def get_db_pages():
    dbManager = DBManager()
    print("Getting DB pages...")
    all_node_pages = dbManager.get_winmodules(ALGORITHM)
    return all_node_pages

if __name__ == "__main__":
    # Your existing code for logging setup and get_db_pages

    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        futures = {}
        for percentage in PERCENTAGE_OF_PAGES:
            for model in MODELS:
                print(f"Executing {percentage}")
                future = executor.submit(perform_benchmark, percentage, model)
                futures[future] = percentage

        print("Waiting for all tasks to complete...")
        concurrent.futures.wait(futures)

# Check: hashTLSH = "T191815C2B7517B0ABCEB6E462159D1B06D0EC1DA7F705B8E02C86F66ED1364E225C194C"