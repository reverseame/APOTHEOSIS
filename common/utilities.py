# common utilities used by more than one script in the project

# https://stackoverflow.com/questions/54366106/configure-formatting-for-root-logger
import sys
import logging
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

import argparse
def configure_argparse() -> argparse.ArgumentParser:
    """Configures argparse to receive HNSW parameters + loglevel."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--M', type=int, default=4, help="Number of established connections of each node (default=4)")
    parser.add_argument('--ef', type=int, default=4, help="Exploration factor (determines the search recall, default=4)")
    parser.add_argument('--Mmax', type=int, default=8, help="Max links allowed per node at any layer, but layer 0 (default=8)")
    parser.add_argument('--Mmax0', type=int, default=16, help="Max links allowed per node at layer 0 (default=16)")
    parser.add_argument('--heuristic', help="Create a HNSW structure using a heuristic to select neighbors rather than a simple selection algorithm (disabled by default)", action='store_true')
    parser.add_argument('--no-extend-candidates', help="Neighbor heuristic selection extendCandidates parameter (enabled by default)", action='store_true')
    parser.add_argument('--no-keep-pruned-conns', help="Neighbor heuristic selection keepPrunedConns parameter (enabled by default)", action='store_true')
    parser.add_argument('--draw', help="Draws the underlying HNSW structure to file (disabled by default)", action='store_true')
    # get log level from command line
    parser.add_argument('-log', '--loglevel', choices=["debug", "info", "warning", "error", "critical"], default='warning', help="Provide logging level (default=warning)")

    return parser
