from hnsw import HNSW
from node_hash import HashNode
from tlsh_algorithm import TLSHHashAlgorithm

# Create an HNSW structure
myHNSW = HNSW(4, ef=4, Mmax=8, Mmax0=16)

# Create the nodes based on TLSH Fuzzy Hashes
node1 = HashNode("T12B81E2134758C0E3CA097B381202C62AC793B46686CD9E2E8F9190EC89C537B5E7AF4C", TLSHHashAlgorithm)
node2 = HashNode("T10381E956C26225F2DAD9D5C2C5C1A337FAF3708A25012B8A1EACDAC00B37D557E0E714", TLSHHashAlgorithm)
node3 = HashNode("T1DF8174A9C2A506F9C6FFC292D6816333FEF1B845C419121A0F91CF5359B5B21FA3A304", TLSHHashAlgorithm)

# Insert nodes on the HNSW structure
myHNSW.add_node(node1)
myHNSW.add_node(node2)
myHNSW.add_node(node3)

# Perform k-nearest neighbor search based on TLSH fuzzy hash similarity
query_node = HashNode("T1BF81A292E336D1F68224D4A4C751A2B3BB353CA9C2103BA69FA4C7908761B50F22E301", TLSHHashAlgorithm)
results = myHNSW.knn_search(query_node, k=5, ef=4)

# Perform percentage search to retrieve nodes above a similarity threshold
results = myHNSW.percentage_search(query_node, percentage=60)

# Dump created HNSW structure to disk
myHNSW.dump("myHNSW.txt")

# Restore HNSW structure from disk
myHNSW = HNSW.load("myHNSW.txt")