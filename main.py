from hnsw import HNSW
from node_number import NumberNode
from node_hash import HashNode
from tlsh_algorithm import TLSHHashAlgorithm



myHNSW = HNSW.load("test.pickle")
print(myHNSW)


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