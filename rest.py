#!/usr/bin/env python
# -*- coding: utf-8 -*-

from flask import Flask, jsonify, request, current_app, g, abort, url_for, redirect
from datetime import datetime
from functools import wraps
from werkzeug.exceptions import HTTPException, InternalServerError
import threading
import uuid
import os
import configparser as cg
import logging
import base64
import json

from datalayer.db_manager import DBManager
from datalayer.hash_algorithm.tlsh_algorithm import TLSHHashAlgorithm
from datalayer.hash_algorithm.ssdeep_algorithm import SSDEEPHashAlgorithm
from datalayer.node.hash_node import HashNode

app = Flask(__name__)

# Dict. to store tasks currently running

tasks = {}

# global vars (important vars)
db_manager = DBManager()
apotheosis_tlsh = None
apotheosis_ssdeep = None

@app.before_request
def before_request():
    """Clean up old tasks. """
    global tasks
    # Keep track of tasks which didn't finish 5 minutes ago
    five_min_ago = datetime.timestamp(datetime.utcnow()) - 5 * 60
    tasks = {task_id: task for task_id, task in tasks.items()
             if ('completion_timestamp' not in task or
                 task['completion_timestamp'] > five_min_ago) or not (task.get('visited', False))}

def get_config_file(path):
    """Reads a configuration file.

    Arguments:
    path -- Path where the configuration file is located
    """
    if os.path.exists(path):
        config = cg.ConfigParser()
        config.read(path)
    else:
        logging.debug(f"Incorrect path provided: {path}")
        print("[-] Config file not found :(, exiting now...")
        exit(1)
    return config

def get_params(params):
    """Returns params of db connection from dict.

    Arguments:
    params -- dict containing info
    """
    return params.get("host", "localhost"), params.get("user", "root"), params.get("pwd"), params.get("dbname")

def is_base64(text):
    try:
        base64.b64decode(text)
        return True
    except base64.binascii.Error:
        return False

def async_api(wrapped_function):
    @wraps(wrapped_function)
    def new_function(*args, **kwargs):
        def task_call(flask_app, environ):
            # Create a request context similar to that of the original request
            # so that the task can have access to flask.g, flask.request, etc.
            with flask_app.request_context(environ):
                try:
                    before = datetime.utcnow()
                    return_value = wrapped_function(*args, **kwargs)
                    response, status_code = return_value
                    after = datetime.utcnow()
                    logging.debug(f"[*] IP {request.remote_addr} requested {request.path} ({status_code}): {response}")
                    logging.debug(f"[*] Elapsed time: {after - before}")
                    tasks[task_id]['return_value'] = return_value
                    
                except HTTPException as e:
                    logging.debug(f"Exception occurred: {e}")
                    tasks[task_id]['return_value'] = current_app.handle_http_exception(e)
                except Exception as e:
                    # The function raised an exception, so we set a 500 error
                    logging.debug(f"Exception occurred: {e}")
                    tasks[task_id]['return_value'] = InternalServerError()
                    raise

                finally:
                    # We record the time of the response, to help in garbage
                    # collecting old tasks
                    tasks[task_id]['completion_timestamp'] = datetime.timestamp(datetime.utcnow())
                    logging.debug(f"Task finished {task_id}")
                    # close the database session (if any)

        # Assign an id to the asynchronous task
        task_id = uuid.uuid4().hex

        # Record the task, and then launch it
        tasks[task_id] = {'task_thread': threading.Thread(
            target=task_call, args=(current_app._get_current_object(),
                                    request.environ))}
        tasks[task_id]['task_thread'].start()
        tasks[task_id]['visited'] = False

        # Return a 202 response, with a link that the client can use to
        # obtain task status
        uri = (url_for('get_task_status', task_id=task_id))
        logging.debug(f"New task {task_id} created: {uri}")
        return redirect(f"{uri}")

    return new_function

@app.route("/status/<string:task_id>/", methods=["GET"])
def get_task_status(task_id):
    """Gets the status of a background task.
    
    Arguments:
    task_id -- id of the task to check status 
    """
    
    logging.debug(f"Asking for TASK ID {task_id} ...")
    task = tasks.get(task_id)
    if task is None:
        abort(404)
    if 'return_value' not in task:
        return 'Your food order is still in the process, please stop by later', 202, {
            'Location': url_for('get_task_status', task_id=task_id)}
    
    logging.debug(f"Task {task_id} finished & visited")
    task['visited'] = True
    return task['return_value']

def _extend_results_winmodule_data(hash_algorithm: str, results: dict) -> dict:
    """Extends the results dict with Winmodule information (from the database).

    Arguments:
    results -- dict of WinModuleHashNode
    """

    new_results = {}
    for key in results:
        if new_results.get(key) is None:
            new_results[key] = {}
        for node in results[key]:
            new_results[key] = node.get_module().as_dict()

    return new_results

def _search_hash(apotheosis_instance, search_type, search_param, hash_algorithm, hash_node: HashNode):
    """Makes a search_type search, with search_params, of the hash node in the given apotheosis instance.
    Returns a JSON with bool 'found' value to indicate if the hash value was found, and 
    a list of 'hashes' with the search results found. 

    Arguments:
    apotheosis_instance -- instance of Apotheosis to use
    search_type         -- search type
    search_param        -- search param
    hash_algorithm      -- hash algorithm
    hash_node           -- hash node to search
    """

    if search_type == "knn":
        found, node, result_dict = apotheosis_instance.knn_search(query=hash_node, k=int(search_param))
    else:
        found, node, result_dict = apotheosis_instance.threshold_search(hash_node, int(search_param), 4)  # Careful this 4!
    
    logging.debug(f"[*] Node \"{hash_node.get_id()}\" {'NOT' if not found else ''} found ({hash_algorithm})")
    
    result_dict = _extend_results_winmodule_data(hash_algorithm, result_dict)
    if node:
        node = db_manager.get_winmodule_data_by_hash(algorithm=hash_algorithm, hash_value=node.get_id())
        node = {key: value for key, value in node.items()}

    logging.debug(f"[*] Found? {found} ({result_dict})")
    result = {"found": found,\
                "query": node,\
                "hashes":
                    {key: value for key, value in result_dict.items()}
                }

    return result

@app.route("/search/<string:search_type>/<int:search_param>/<string:hash_algorithm>/<path:hash_value>/", methods=["GET"])
@async_api
def search(search_type, search_param, hash_algorithm, hash_value):
    """Perform a search_type, using search_param, of the hash_value (from hash_algorithm) in Apotheosis.
    Returns a JSON response (base64 encoded) containing the search results.

    Arguments:
    search_type    -- type of search ("knn" or "threshold")
    search_param   -- search parameter for the search_type
    hash_algorithm -- distance algorithm ("tlsh" or "ssdeep")
    hash_value     -- hash to search (base64 encoded)
    """
    
    validation_error = _validate_parameters(search_type, hash_algorithm)
    if validation_error:
        return validation_error

    hash_algorithm_class = TLSHHashAlgorithm if hash_algorithm == "tlsh" else SSDEEPHashAlgorithm
    apotheosis_instance = apotheosis_tlsh if hash_algorithm == "tlsh" else apotheosis_ssdeep

    # decode input (it comes in base64)
    try:
        hash_value = base64.b64decode(hash_value).decode('utf-8')
    except Exception as e:
        logging.error(f"Decoding error {e.args} with {hash_value}")
        msg = base64.b64encode(f"Error processing \"{hash_value}\". Contact an admin")
        return msg, InternalServerError()
    hash_node = HashNode(hash_value, hash_algorithm_class)

    logging.debug(f"Simple search of {hash_value} ({search_type} {search_param} in {hash_algorithm}")
    json_result = _search_hash(apotheosis_instance, search_type, search_param, hash_algorithm, hash_node)  
    return_value = base64.b64encode(str.encode(str(json_result)))

    return return_value, 200

def _validate_parameters(search_type, hash_algorithm):
    """Validates search parameters.
    Returns None on success. Otherwise, returns a tuple of str and int

    Arguments:
    search_type     -- supported search type
    hash_algorithm  -- supported hash algorithm
    """

    logging.debug(f"Validating {search_type} and {hash_algorithm} ...")
    supported_search_types = ["knn", "threshold"]
    supported_hash_algorithms = ["tlsh", "ssdeep"]

    if search_type not in supported_search_types:
        logging.debug(f"Search algorithm unsupported: {search_type}")
        return f"The search algorithm {search_type} is not supported (expected values: ', '.join(supported_search_types))", 400

    if hash_algorithm not in supported_hash_algorithms:
        logging.debug(f"Hash algorithm unsupported: {hash_algorithm}")
        return f"The hash algorithm {hash_algorithm} is not supported {', '.join(supported_hash_algorithms)}", 400

    return None

@app.route("/bulk/<string:hash_algorithm>/<string:search_type>/<int:search_param>/", methods=["POST"])
@async_api
def bulk_search(hash_algorithm, search_type, search_param):
    """Performs an Apotheosis search to multiple hashes (they come by POST, base64 encoded).
    Returns a JSON response (base64 encoded) containing the search results for each hash.

    Arguments:
    hash_algorithm -- hash algorithm
    search_type    -- type of search
    search_param   -- search parameter
    """

    if not request.is_json:
        logging.debug(f"POST and not JSON request: {request}")
        return "You can only post JSON data, son", 400

    validation_error = _validate_parameters(search_type, hash_algorithm)
    if validation_error:
        return validation_error

    try:
        hashes = request.get_json()['hashes']
        if len(hashes) == 0:
            logging.debug(f"Hash list is empty!")
            return "Nothing to query: hash list is empty", 400
    except KeyError:
        logging.debug(f"Bad JSON POST: {request.get_json()}")
        return "Bad JSON POST", 400

    hash_algorithm_class = TLSHHashAlgorithm if hash_algorithm == "tlsh" else SSDEEPHashAlgorithm
    apotheosis_instance  = apotheosis_tlsh if hash_algorithm == "tlsh" else apotheosis_ssdeep

    logging.debug(f"Bulk {search_type} search with {search_param} ({hash_algorithm})")
    result_list = []
    for hash_value in hashes:
        # decode input (it comes in base64)
        try:
            hash_value = base64.b64decode(hash_value).decode('utf-8')
        except Exception as e:
            logging.error(f"Encoding error {e.args} with {hash_value}")
            pass
        hash_node = HashNode(hash_value, hash_algorithm_class)
        # get JSON results and append to result list
        json_result = _search_hash(apotheosis_instance, search_type, search_param, hash_algorithm, hash_node)  
        result_list.append(json_result)

    if len(result_list) == 0:
        return "Error processing your bulk request. Contact an admin", 500

    json_result_list = json.dumps(result_list)
    # encode and return them
    return_value = base64.b64encode(str.encode(str(json_result_list)))
    return return_value, 200

# just for testing
def load_apotheosis(apo_model_tlsh: str=None, apo_model_ssdeep: str=None,
                        args=None):
    global apotheosis_tlsh
    global apotheosis_ssdeep

    from apotheosis import Apotheosis # avoid circular deps

    if args is None:
        print("[*] Loading Apotheosis model with TLSH ...")
        apotheosis_tlsh = Apotheosis.load(filename=apo_model_tlsh, distance_algorithm=TLSHHashAlgorithm)
        
        if apo_model_ssdeep:
            print("[*] Loading Apotheosis with SSDEEP ...")
            apotheosis_ssdeep = Apotheosis.load(filename=apo_model_ssdeep,\
                                        distance_algorithm=SSDEEPHashAlgorithm)
    else:
        apotheosis_tlsh = Apotheosis(M=args.M, ef=args.ef, Mmax=args.Mmax, Mmax0=args.Mmax0,\
                    heuristic=args.heuristic,\
                    extend_candidates=False, keep_pruned_conns=False,\
                    beer_factor=0,\
                    distance_algorithm=TLSHHashAlgorithm)
        
        # load from DB and insert into the model
        print("[*] Building Apotheosis with TLSH ...")
        utils.load_DB_in_model(npages=args.npages, algorithm=TLSHHashAlgorithm, current_model=apotheosis_tlsh)
    
        apotheosis_ssdeep = apotheosis_tlsh

import sys
import common.utilities as utils
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-algorithm', '--distance-algorithm', choices=["tlsh", "ssdeep"], default='tlsh', help="Distance algorithm to be used in the underlying HNSW structure (default=tlsh)")
    parser.add_argument("--port", type=int, default=5000, help="Port to serve (default 5000)")
    parser.add_argument('-f', '--file', type=str, help='Load previously saved APOTHEOSIS model from file')
    parser.add_argument('--npages', type=int, default=None, help="Number of pages to test (default=None -- means all)")
    parser.add_argument('--debug-mode', action='store_true', help="Run REST API in dev mode")
    parser.add_argument('-log', '--loglevel', choices=["debug", "info", "warning", "error", "critical"], default='info', help="Provide logging level (default=warning)")
    parser.add_argument('--M', type=int, default=4, help="Number of established connections of each node (default=4)")
    parser.add_argument('--ef', type=int, default=4, help="Exploration factor (determines the search recall, default=4)")
    parser.add_argument('--Mmax', type=int, default=8, help="Max links allowed per node at any layer, but layer 0 (default=8)")
    parser.add_argument('--Mmax0', type=int, default=16, help="Max links allowed per node at layer 0 (default=16)")
    parser.add_argument('--heuristic', help="Create the underlying HNSW structure using a heuristic to select neighbors rather than a simple selection algorithm (disabled by default)", action='store_true')


    args = parser.parse_args()
    
    log_level = args.loglevel.upper()
    utils.configure_logging(log_level)
    logging.basicConfig(stream=sys.stdout, level=log_level)

    if args.file:
        load_apotheosis(apo_model_tls=args.file)
    else:
        load_apotheosis(args=args)

    print(f"[*] Serving REST API at :{args.port} ... ")
    if args.debug_mode:
        print("[DEBUG MODE]")
        debug = log_level == "DEBUG"
        app.run(debug=debug, host="0.0.0.0", port=args.port)
    else:
        from waitress import serve
        serve(app, host="0.0.0.0", port=args.port)
