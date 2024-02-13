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

from datalayer.db_manager import DBManager
from datalayer.hash_algorithm.tlsh_algorithm import TLSHHashAlgorithm
from datalayer.hash_algorithm.ssdeep_algorithm import SSDEEPHashAlgorithm
from apotheosis import Apotheosis
from datalayer.node.hash_node import HashNode

app = Flask(__name__)

# Dict. to store tasks currently running
tasks = {}

# global vars (important vars)
db_manager = None
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
                    logging.debug(f"[*] IP {request.remote_addr} requested {request.path} ({status_code}): {response.decode(encoding='utf-8')}")
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
            new_results[key] = db_manager.get_winmodule_data_by_hash(hash_algorithm, node.get_id())

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
        found, result_dict = apotheosis_instance.knn_search(hash_node, int(search_param))
    else:
        found, result_dict = apotheosis_instance.threshold_search(hash_node, int(search_param), 4)  # Careful this 4!
    
    logging.debug(f"[*] Node \"{hash_node.get_id()}\" {'NOT' if not found else ''} found ({hash_algorithm})")

    result_dict = _extend_results_winmodule_data(hash_algorithm, result_dict)
    logging.debug(f"[*] Found? {found} ({result_dict})")
    json_result = {'found': found, 'hashes':
                   {key: value for key, value in result_dict.items()}
                   }

    return json_result

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
    hash_value = base64.b64decode(hash_value).decode('utf-8')
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
    except KeyError:
        logging.debug(f"Bad JSON POST: {request.get_json()}")
        return "Bad JSON POST", 400

    hash_algorithm_class = TLSHHashAlgorithm if hash_algorithm == "tlsh" else SSDEEPHashAlgorithm
    apotheosis_instance  = apotheosis_tlsh if hash_algorithm == "tlsh" else apotheosis_ssdeep

    logging.debug(f"Bulk {search_type} search with {search_param} ({hash_algorithm})")
    result_list = []
    for hash_value in hashes:
        # decode input (it comes in base64)
        hash_value = base64.b64decode(hash_value).decode('utf-8')
        hash_node = HashNode(hash_value, hash_algorithm_class)
        # get JSON results and append to result list
        json_result = _search_hash(apotheosis_instance, search_type, search_param, hash_algorithm, hash_node)  
        result_list.append(json_result)

    # encode and return them
    return_value = base64.b64encode(str.encode(str(result_list)))
    return return_value

# just for testing
def load_apotheosis(apo_model_tlsh, apo_model_ssdeep: str=None, db_manager=None):
    global apotheosis_tlsh
    global apotheosis_ssdeep

    apotheosis_tlsh = Apotheosis.load(apo_model_tlsh, distance_algorithm=TLSHHashAlgorithm, db_manager=db_manager)
    if apo_model_ssdeep:
        apotheosis_ssdeep = Apotheosis.load(apo_model_ssdeep, distance_algorithm=SSDEEPHashAlgorithm, db_manager=db_manager)

import sys
import common.utilities as utils
import requests
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    parser.add_argument('-log', '--loglevel', choices=["debug", "info", "warning", "error", "critical"], default='warning', help="Provide logging level (default=warning)")
   
    args = parser.parse_args()
    
    log_level = args.loglevel.upper()
    utils.configure_logging(log_level)

    logging.basicConfig(stream=sys.stdout, level=log_level)

    db_manager = DBManager()
    load_apotheosis(args.filename, db_manager=db_manager)
    debug= log_level == "DEBUG"
    app.run(debug=debug, host="0.0.0.0")
