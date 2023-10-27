#!/usr/bin/env python
# -*- coding: utf-8 -*-

from flask import Flask, jsonify, request, current_app, g, abort, url_for, redirect
from mysql.connector import connect, Error
from datetime import datetime
from functools import wraps
from werkzeug.exceptions import HTTPException, InternalServerError
import time
from subprocess import check_output
from db_manager2 import DBManager
from tlsh_algorithm import TLSHHashAlgorithm
from hnsw import HNSW
from node_hash import HashNode
import threading
import uuid
import json
import sys
import os
import configparser as cg
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
app = Flask(__name__)
# Dict. to store tasks currently running
tasks = {}
# Dict. to store parameters to log server
param = {}

HNSW_HASH_ALGORITHM = TLSHHashAlgorithm

db_manager = None
hnsw = None

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

    Args:
        path (string): Path where the configuration file is located

    Returns:
        dict : Returns a configuration file fields
    """
    if os.path.exists(path):
        config = cg.ConfigParser()
        config.read(path)
    else:
        print("[-] Config file not found :(, exiting now...")
        sys.exit(1)
    return config


def get_params(params):
    """Returns params of db connection from dict.

    Args:
        params (dict): dict containing info

    Returns:
        tuple: parameters
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
                    if current_app.debug:
                        # We want to find out if something happened so raise
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
def gettaskstatus(task_id):
    task = tasks.get(task_id)
    if task is None:
        abort(404)
    if 'return_value' not in task:
        return 'The job is still being processed, please refresh later', 202, {
            'Location': url_for('gettaskstatus', task_id=task_id)}
    task['Visited'] = True
    return task['return_value']


@app.route("/search/<string:search_type>/<string:search_param>/<path:hash_value>/", methods=["GET"])
def get_hash(search_type, search_param, hash_value):
    """Perform HNSW search to a hash"""
    if search_type == "knn":
        hashes =  hnsw.knn_search(HashNode(hash_value, HNSW_HASH_ALGORITHM), int(search_param), hnsw.ef)
    elif search_type == "threshold":
        hashes = hnsw.percentage_search(HashNode(hash_value, HNSW_HASH_ALGORITHM), int(search_param))
    else:
        return "The search algorithm supplied is not supported", 400

    modules = []
    for hash in hashes:
        modules.append(hash.module.as_dict())

    return modules, 200


@app.route("/bulk/<string:search_type>/<string:search_param>/", methods=["POST"])
@async_api
def bulk_hash(search_type, search_param):
    """Perform HNSW search to a hash"""
    if not request.is_json:
        return "You can only post JSON data", 400
    if search_type not in ["knn", "thershold"]:
        return "The search algorithm supplied is not supported", 400
    
    data_received_in_request = request.get_json()
    try:
        hashes = data_received_in_request["hashes"]
    except:
        return "Bad JSON POST", 400

    res = []
    for h in hashes:
        modules = []
        if search_type == "knn":
            results =  hnsw.knn_search(HashNode(h, HNSW_HASH_ALGORITHM), int(search_param), hnsw.ef)
        elif search_type == "threshold":
            results = hnsw.percentage_search(HashNode(h, HNSW_HASH_ALGORITHM), int(search_param))
        
        for result in results:
            modules.append(result.module.as_dict())

        res.append({h: modules})
    print(res)
    return res, 200

def initialize_hnsw():
    global hnsw
    logger.info("Getting pages from DB...")
    list_pages = db_manager.get_winmodules(HNSW_HASH_ALGORITHM)
    logger.info("Creating HNSW model...")
    hnsw = HNSW(64, 64, 128, 256)
    for page in list_pages[0:100]:
        hnsw.add_node(page)

if __name__ == "__main__":
    db_manager = DBManager()
    initialize_hnsw()
    app.run(debug=False, host="0.0.0.0")