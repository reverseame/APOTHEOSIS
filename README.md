# APOTHEOSIS

APOTHEOSIS (*APprOximaTe searcH systEm Of Similarity dIgeSts*) is a powerful system to perform similarity search, using Radix Tree and specialized implementation of the Hierarchical Navigable Small World (HNSW) data structure adapted for efficient nearest neighbor lookup of approximate matching hashes.

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

## Features
- Construction of APOTHEOSIS model, consisting on two data structures: Radix Tree and HNSW.
- Insertion of nodes in the system.
- K-nearest neighbor search based on similarity.
- Threshold search to retrieve nodes that meet a similarity threshold.
- Several scripts to test and benchmark the system and generate graphs.
- Logging functionality for debugging and monitoring.
- Ability to save to disk and restore the index structure.
- REST API for easily adaption on existing workflows

## Installation
You can install the necessary dependencies using:
```
pip install -r requirements.txt
```

# System configuration parameters
In order to reach the proper balance between precission and speed, some configuration values can be modified in order to tune the performance. This configuration values have impact on HNSW data structure mainly. Values may be adjusted depending on your use case.

- *M*: specifies the number of connections of a new node when inserted.
- *Mmax*: specifies the maximum number of neighbors (connections) a node can have at each layer of the hierarchy higher than zero
- *Mmax0*: specifies the maximum number of neighbors (connections) a node can have at each layer of the hierarchy at layer zero
- *ef*: : controls the number of neighbors to explore during the construction and search phase of the HNSW graph.
- *heuristic* (True/False):  Perform KNN search with heuristic. May improve performance, check Algorithm 4 in MY-TPAMI-20 for more information. 
- *keep_pruned_conns* (True/False):  Indicate whether or not to add discarded elements when performing heuristic search.
- *beer_factor*: Performs random walk when exploring the neighborhood. 


## Usage

```python
from apotheosis_winmodule import ApotheosisWinModule
from datalayer.node import HashNode
from datalayer.hash_algorithm.tlsh_algorithm import TLSHHashAlgorithm

# Create an APOTHEOSIS model
myApo = ApotheosisWinModule(M=4, ef=4, Mmax=8, Mmax0=16,\
                    heuristic=False,\
                    extend_candidates=False, keep_pruned_conns=False,\
                    beer_factor=0,\
                    distance_algorithm=TLSHHashAlgorithm)

# Create the nodes based on TLSH algorithm approximate matching hashes
node1 = HashNode("T12B81E2134758C0E3CA097B381202C62AC793B46686CD9E2E8F9190EC89C537B5E7AF4C", TLSHHashAlgorithm)
node2 = HashNode("T10381E956C26225F2DAD9D5C2C5C1A337FAF3708A25012B8A1EACDAC00B37D557E0E714", TLSHHashAlgorithm)
node3 = HashNode("T1DF8174A9C2A506F9C6FFC292D6816333FEF1B845C419121A0F91CF5359B5B21FA3A304", TLSHHashAlgorithm)

# Insert nodes on the APOTHEOSIS system
myApo.insert(node1)
myApo.insert(node2)
myApo.insert(node3)

# Perform k-nearest neighbor search based on TLSH approximate matching hashes
query_node = HashNode("T1BF81A292E336D1F68224D4A4C751A2B3BB353CA9C2103BA69FA4C7908761B50F22E301", TLSHHashAlgorithm)
results = myApo.knn_search(query_node, k=5)

# Perform threshold search to retrieve nodes above a similarity threshold
results = myApo.threshold_search(query_node, threshold=60)

# Dump created APOTHEOSIS structure to disk
myHNSW.dump("myAPO")
```

## License
Licensed under the [GNU GPLv3](LICENSE) license.
