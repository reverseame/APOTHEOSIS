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

from db_manager import DBManager
from datalayer.hash_algorithm.tlsh_algorithm import TLSHHashAlgorithm
from datalayer.hash_algorithm.ssdeep_algorithm import SSDEEPHashAlgorithm
from apotheosis import Apotheosis
from datalayer.node.hash_node import HashNode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Dict. to store tasks currently running
tasks = {}

db_manager = None
apotheosis_tlsh = None
apotheosis_ssdeep = None

@app.before_request
def before_request():
    """Clean up old tasks"""
    global tasks
    # Keep track of tasks which didn't finish 5 minutes ago
    five_min_ago = datetime.timestamp(datetime.utcnow()) - 5 * 60
    tasks = {task_id: task for task_id, task in tasks.items()
             if ('completion_timestamp' not in task or
                 task['completion_timestamp'] > five_min_ago) or not (task.get('Visited', False))}

def get_config_file(path):
    """Reads a configuration file

    Arguments:
    path -- Path where the configuration file is located

    """
    if os.path.exists(path):
        config = cg.ConfigParser()
        config.read(path)
    else:
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
                    tasks[task_id]['return_value'] = wrapped_function(*args, **kwargs)
                except HTTPException as e:
                    tasks[task_id]['return_value'] = current_app.handle_http_exception(e)
                except Exception:
                    # The function raised an exception, so we set a 500 error
                    tasks[task_id]['return_value'] = InternalServerError()
                    raise

                finally:
                    # We record the time of the response, to help in garbage
                    # collecting old tasks
                    tasks[task_id]['completion_timestamp'] = datetime.timestamp(datetime.utcnow())
                    print("Job finished")
                    # close the database session (if any)

        # Assign an id to the asynchronous task
        task_id = uuid.uuid4().hex

        # Record the task, and then launch it
        tasks[task_id] = {'task_thread': threading.Thread(
            target=task_call, args=(current_app._get_current_object(),
                                    request.environ))}
        tasks[task_id]['task_thread'].start()
        tasks[task_id]['Visited'] = False

        # Return a 202 response, with a link that the client can use to
        # obtain task status
        uri = (url_for('gettaskstatus', task_id=task_id))
        print(uri)
        return redirect(f"{uri}")

    return new_function

@app.route("/status/<string:task_id>/", methods=["GET"])
def get_task_status(task_id):
    """Get the status of a background task
    
    Arguments:
    task_id -- id of the task to check status 
    """
    task = tasks.get(task_id)
    if task is None:
        abort(404)
    if 'return_value' not in task:
        return 'The job is still being processed, please refresh later', 202, {
            'Location': url_for('gettaskstatus', task_id=task_id)}
    task['Visited'] = True
    return task['return_value']

@app.route("/search/<string:hash_algorithm>/<string:search_type>/<int:search_param>/<path:hash_value>/", methods=["GET"])
def simple_search(hash_algorithm, search_type, search_param, hash_value):
    """Perform a search on the 

    Arguments:
    hash_algorithm -- The distance algorithm ("tlsh" or "ssdeep").
    search_type    -- The type of search ("knn" or "threshold").
    search_param   -- The search parameter for the search_type.
    hash_value     -- The hash to search.

    Returns:
    response: JSON response containing the search results.
    """
    validation_error = _validate_parameters(search_type, hash_algorithm)
    if validation_error:
        return validation_error

    hash_algorithm_class = TLSHHashAlgorithm if hash_algorithm == "tlsh" else SSDEEPHashAlgorithm
    apotheosis_instance = apotheosis_tlsh if hash_algorithm == "tlsh" else apotheosis_ssdeep

    if search_type == "knn":
        found, result_dict = apotheosis_instance.knn_search(HashNode(hash_value, hash_algorithm_class), int(search_param))
    else:
        found, result_dict = apotheosis_instance.threshold_search(HashNode(hash_value, hash_algorithm_class), int(search_param), 4)  # Careful this 4!

    json_result = {'found': found, 'hashes':
                   {key: [hash._module.as_dict() for hash in value] for key, value in result_dict.items()}
                   }

    return jsonify(json_result), 200

def _validate_parameters(search_type, hash_algorithm):
    """Validate search parameters"""

    supported_search_types = ["knn", "threshold"]
    supported_hash_algorithms = ["tlsh", "ssdeep"]

    if search_type not in supported_search_types:
        return "The search algorithm supplied is not supported", 400

    if hash_algorithm not in supported_hash_algorithms:
        return "The hash algorithm supplied is not supported", 400

    return None

@app.route("/bulk/<string:hash_algorithm>/<string:search_type>/<int:search_param>/", methods=["POST"])
@async_api
def bulk_search(hash_algorithm, search_type, search_param):
    """Perform Apotheosis search to multiple hashes

    Args:
        hash_algorithm -- The hash algorithm ("tlsh" or "ssdeep").
        search_type    -- The type of search ("knn" or "threshold").
        search_param   -- The search parameter.

    Returns:
        response: JSON response containing the search results for each hash.
    """

    if not request.is_json:
        return "You can only post JSON data", 400

    validation_error = _validate_parameters(search_type, hash_algorithm)
    if validation_error:
        return validation_error

    try:
        hashes = request.get_json()['hashes']
    except KeyError:
        return "Bad JSON POST", 400

    hash_algorithm_class = TLSHHashAlgorithm if hash_algorithm == "tlsh" else SSDEEPHashAlgorithm
    apotheosis_instance = apotheosis_tlsh if hash_algorithm == "tlsh" else apotheosis_ssdeep

    result_list = []
    for hash_value in hashes:
        hash_node = HashNode(hash_value, hash_algorithm_class)
        if search_type == "knn":
            found, result_dict = apotheosis_instance.knn_search(hash_node, int(search_param))
        else:
            found, result_dict = apotheosis_instance.threshold_search(hash_node, int(search_param), 4)  # Careful this 4!

        json_result = {'found': found, 'hashes':
                        {key: [hash._module.as_dict() for hash in value] for key, value in result_dict.items()}
                      }

        result_list.append(json_result)

    return result_list

def _load_apotheosis():
    global apotheosis_tlsh
    global apotheosis_ssdeep

    apotheosis_tlsh = Apotheosis.load("rest_model")  # Script parameter?
    #apotheosis_ssdeep = Apotheosis.load("rest_model_ssdeep")

import requests
if __name__ == "__main__":
    db_manager = DBManager()
    _load_apotheosis()
    app.run(debug=False, host="0.0.0.0")

    # Some tests
    base_url = "http://localhost:5000"
    search_url = f"{base_url}/search/knn/5/tlsh/T12B81E2134758C0E3CA097B381202C62AC793B46686CD9E2E8F9190EC89C537B5E7AF4C"
    response = requests.get(search_url)
    print(response.json())
