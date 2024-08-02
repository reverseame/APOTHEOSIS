import os
import json
import argparse
import logging
import traceback
from apotheosis import Apotheosis
from datetime import datetime
from datalayer.node.chapbook_node import ChapBookHashNode
from datalayer.hash_algorithm.tlsh_algorithm import TLSHHashAlgorithm
from datalayer.hash_algorithm.ssdeep_algorithm import SSDEEPHashAlgorithm

def configure_argparse():
    parser = argparse.ArgumentParser(description='Configure and manage Apotheosis structures for different hash algorithms.')
    parser.add_argument('--directory', type=str, required=True, help='Directory containing JSON files.')
    parser.add_argument('--M', type=int, default=4, help='Number of connections of a new node when inserted.')
    parser.add_argument('--ef', type=int, default=4, help='Controls the number of neighbors to explore during the construction and search phase of the HNSW graph')
    parser.add_argument('--Mmax', type=int, default=8, help='Specifies the maximum number of neighbors (connections) a node can have at each layer of the hierarchy higher than zero')
    parser.add_argument('--Mmax0', type=int, default=16, help='Specifies the maximum number of neighbors (connections) a node can have at each layer of the hierarchy at layer zero')
    parser.add_argument('--heuristic', default=False, action='store_true', help='Enable heuristic mode')
    parser.add_argument('--no_extend_candidates', action='store_true', help='Disable extending candidates during search')
    parser.add_argument('--no_keep_pruned_conns', action='store_true', help='Do not keep pruned connections')
    parser.add_argument('--beer_factor', type=float, default=0, help='Performs random walk when exploring the neighborhood')
    parser.add_argument('--loglevel', type=str, default='ERROR', help='Set the logging level')
    return parser

import tlsh
def compute_distance_tlsh(hash1, hash2):
    """
    Compute the TLSH distance between two hashes.
    """
    if hash1 and hash2:
        return tlsh.diff(hash1, hash2)
    return None

import ssdeep
def compute_distance_ssdeep(hash1, hash2):
    """
    Compute the SSDEEP distance between two hashes.
    """
    if hash1 and hash2:
        return ssdeep.compare(hash1, hash2)
    return None

def compute_matrix(chapters, is_tlsh: bool=True):
    """
    Compute a distance matrix for all pairs of chapters.
    """
    nodes = list(sorted(chapters.keys()))
    distance_matrix = {}

    for i, node1 in enumerate(nodes):
        for j, node2 in enumerate(nodes):
            if i < j:  # Avoid duplicate computations, as distance is symmetric
                if is_tlsh:
                    distance = compute_distance_tlsh(chapters[node1], chapters[node2])
                else:
                    distance = compute_distance_ssdeep(chapters[node1], chapters[node2])
                distance_matrix[(node1, node2)] = distance

    table = generate_latex_table(nodes, distance_matrix)
    print(table)

    return distance_matrix

def generate_latex_table(nodes, distance_matrix):
    """
    Generate a LaTeX table from the distance matrix.
    """
    table = "\\begin{tabular}{l" + "c" * len(nodes) + "}\n"
    table += " & {\\bf " + "} & {\\bf ".join(nodes) + " \\\\\n"
    table += "\\hline\n"

    for i, node1 in enumerate(nodes):
        row = ["{\\bf " + node1 + "}"]
        for j, node2 in enumerate(nodes):
            if i == j:
                row.append("-")  # Diagonal (same chapter comparison)
            elif (node1, node2) in distance_matrix:
                row.append(str(distance_matrix[(node1, node2)]))
            elif (node2, node1) in distance_matrix:
                row.append(str(distance_matrix[(node2, node1)]))
            else:
                row.append("")

        table += " & ".join(row) + " \\\\\n"

    table += "\\end{tabular}\n"
    return table


def insert_node(apotheosis_instance, hash_algorithm, filename, _hash):
    node = ChapBookHashNode(
            id=_hash,
            hash_algorithm=hash_algorithm,
            filename=filename,
        )
    try:
        apotheosis_instance.insert(node)
        logging.info(f"Inserted node: {node._id} with algorithm: {hash_algorithm.__name__}")
    except Exception as e:
        logging.error(f"Failed to insert node: {node._id} with algorithm: {hash_algorithm.__name__} - Error: {str(e)}")
        logging.error(traceback.format_exc())
    
    return apotheosis_instance

def main():
    parser = configure_argparse()
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(level=getattr(logging, args.loglevel.upper()), format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Starting Apotheosis script")

    # Create instances of Apotheosis for each hash algorithm
    logging.info("Creating Apotheosis instances")
    apo_tlsh = Apotheosis(distance_algorithm=TLSHHashAlgorithm, M=args.M, ef=args.ef, Mmax=args.Mmax, Mmax0=args.Mmax0,
                          heuristic=not args.heuristic, extend_candidates=not args.no_extend_candidates,
                          keep_pruned_conns=not args.no_keep_pruned_conns, beer_factor=args.beer_factor)
    apo_ssdeep = Apotheosis(distance_algorithm=SSDEEPHashAlgorithm, M=args.M, ef=args.ef, Mmax=args.Mmax, Mmax0=args.Mmax0,
                            heuristic=args.heuristic, extend_candidates=not args.no_extend_candidates,
                            keep_pruned_conns=not args.no_keep_pruned_conns, beer_factor=args.beer_factor)

    directory_path = args.directory
    logging.info(f"Processing JSON files in directory: {directory_path}")

    _dict_tlsh = {}    
    _dict_ssdeep = {}    
    for filename in os.listdir(directory_path):
        if filename.endswith(".json"):
            file_path = os.path.join(directory_path, filename)

            logging.info(f"Loading JSON file: {file_path}")
            with open(file_path, 'r') as file:
                data = json.load(file)
                for filename, hashes in data.items():
                    tlsh, ssdeep = hashes
                    insert_node(apo_tlsh, TLSHHashAlgorithm, filename, tlsh)
                    insert_node(apo_ssdeep, SSDEEPHashAlgorithm, filename, ssdeep)
                    _min = filename.split('_')[1]
                    _max = filename.split('_')[3].split('.')[0]
                    _str = f"{int(_min):02d}-{int(_max):02d}"
                    _dict_tlsh.update({ _str: tlsh })
                    _dict_ssdeep.update({ _str: ssdeep })
    
    compute_matrix(_dict_tlsh)
    compute_matrix(_dict_ssdeep, is_tlsh=False)

    apo_tlsh.draw(filename="_TLSH", threshold=0)
    apo_ssdeep.draw(filename="_ssdeep", threshold=0)

if __name__ == "__main__":
    main()

