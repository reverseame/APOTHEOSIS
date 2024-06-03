#TODO docstring
import json
import os
from datalayer.node.hash_node import HashNode
from datalayer.hash_algorithm.hash_algorithm import HashAlgorithm


class IotHashNode(HashNode):
    def __init__(self, id, hash_algorithm: HashAlgorithm, name, size,  file):
        super().__init__(id, hash_algorithm)
        self._name = name
        self._size = size
        self._file = file
        self._family_name = ""
        self._category = self.classify_function()

    def classify_function(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, "functions.json")
        current_candidate = ""
        with open(file_path, 'r') as file:
            functions = json.load(file)
            for function in functions:
                if self._name in function or function in self._name:
                    if len(functions[function]) > len(current_candidate):
                        current_candidate = functions[function]
        
        if current_candidate == "":
            return "Undefined"
        return current_candidate

    def get_name(self):
        return self._name
    
    def get_size(self):
        return self._size
    
    def get_category(self):
        return self._category
    
    def get_file(self):
        return self._file
    
    def get_draw_features(self):
        return {"names": {self._id: self._name.replace(":", "")},
                "sizes": {self._id: self._size.replace(":", "")}, 
                "categories": {self._id: self._category.replace(":", "")}, 
                "files" : {self._id: self._file.replace(":", "")},
                "faimly_names" : {self._id: self._family_name.replace(":", "")}
                }

    def is_equal(self, other):
        return self._name == other._name

