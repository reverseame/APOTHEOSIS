from hnsw import HNSW
from node_number import NumberNode
from node_hash import HashNode
from tlsh_algorithm import TLSHHashAlgorithm
from db_manager import DBManager
import time
import random
dbManager = DBManager()


# CONFIG

models = [(16,32,32,64), (32, 64, 64, 128), 
          (64, 64, 128, 256), (128, 128, 256, 512),
          (100, 100, 100, 200), (200, 200, 200, 400)]

percentajes_of_pages = [0.03, 0.10, 0.25, 0.5, 0.75, 1]
algorithm = TLSHHashAlgorithm

print("Getting pages from DB...")
list_pages = dbManager.get_all_pages()

def insert_nodes(model, n):
    add_times = []
    for i in range(0, n):
        init_time = time.time()
        model.add_node(HashNode(list_pages[i].hashTLSH, algorithm))
        end_time = time.time()
        add_times.append(end_time - init_time)
        print(f"\rAdding {i}/{n} pages", end='', flush=True)
    write_add_times(add_times, model, n)

def search_node(model, hash):
     init_time = time.time()
     model.knn_search(HashNode(hash, algorithm), 1, model.ef)
     return time.time() - init_time
     
def search_nodes(model, n, hashes):    
    search_times = []
    n_hashes = len(hashes)
    for i in range(0,n_hashes):
        print(f"\rSearching hash {i}/{n}", end='', flush=True)
        search_times.append(search_node(model, hashes[i].hashTLSH))
    write_search_times(search_times, model, n)

def write_search_times(search_times, model, n):
    with open(f'model_search_{model.M}_{model.ef}_{model.Mmax}_{model.Mmax0}_{n}.txt', 'w') as f:
        for item in search_times:
            f.write("%s\n" % item)

def write_add_times(add_times, model, n):
    with open(f'model_insert_{model.M}_{model.ef}_{model.Mmax}_{model.Mmax0}_{n}.txt', 'w') as f:
        for item in add_times:
            f.write("%s\n" % item)

for percentaje in percentajes_of_pages:
    n_pages = int(percentaje*len(list_pages))
    hashes = random.sample(list_pages[:n_pages], 100)
    for model in models:
        print("Benchmarking model ({}, {}, {}, {}) with {} pages...".format(*model, n_pages))
        current_model = HNSW(*model)
        insert_nodes(current_model, n_pages)
        search_nodes(current_model, n_pages, hashes)
        

















#myHNSW = HNSW.load("test.pickle")
#print(myHNSW)

'''
myHNSW = HNSW(M=3, ef=3, Mmax=3, Mmax0=5)
myHNSW.add_node(HashNode("T12B81E2134758C0E3CA097B381202C62AC793B46686CD9E2E8F9190EC89C537B5E7AF4C", TLSHHashAlgorithm))
myHNSW.add_node(HashNode("T10381E956C26225F2DAD9D5C2C5C1A337FAF3708A25012B8A1EACDAC00B37D557E0E714", TLSHHashAlgorithm))
myHNSW.add_node(HashNode("T1DF8174A9C2A506F9C6FFC292D6816333FEF1B845C419121A0F91CF5359B5B21FA3A304", TLSHHashAlgorithm))
myHNSW.add_node(HashNode("T1BF81A292E336D1F68224D4A4C751A2B3BB353CA9C2103BA69FA4C7908761B50F22E301", TLSHHashAlgorithm))
print(myHNSW)

myHNSW.dump("test.pickle")
numbers = np.random.uniform(0, 1, size=1000)
myHNSW = HNSW(M=16, ef=32, Mmax=32, Mmax0=32)
for n in numbers:
    myHNSW.add_node(Node(n))

print_HNSW(myHNSW)
result = myHNSW.knn_search(Node(1), 2, 3)
print(f"NN: {[n.id for n in result]}")

myHNSW.add_node(Node(1))
myHNSW.add_node(Node(2))
myHNSW.add_node(Node(4))
myHNSW.add_node(Node(5))
myHNSW.add_node(Node(6))
print_HNSW()
print("-------------")
result = myHNSW.knn_search(Node(2), 2, 3)
print(f"NN: {[n.id for n in result]}")
#myHNSW.add_node(Node(9))
#myHNSW.add_node(Node(10))

'''