#TODO docstring
from datalayer.node.hash_node import HashNode
from datalayer.hash_algorithm.hash_algorithm import HashAlgorithm
from common.constants import *
from common.errors import NodeUnsupportedAlgorithm

class ChapBookHashNode(HashNode):
    def __init__(self, id, hash_algorithm: HashAlgorithm, filename: str=None):
        super().__init__(id, hash_algorithm)
        self._filename = filename

    def __lt__(self, other): # Hack for priority queue. TODO: not needed here?
        return False

    def get_filename(self):
        return self._filename

    #XXX having : as character in a node label is not allowed, trick to avoid it
    def _sanitize_id(self):
        return self._id.replace(':', '')

    def get_draw_features(self):
        name = self.get_filename().split('.')[0]
        _min = name.split('_')[1]
        _max = name.split('_')[3]
        return {"filename": { self._sanitize_id(): self.get_filename()},
                "label": { self._sanitize_id(): f"{_min}-{_max}" },
                "caps": { self._sanitize_id(): f"{_min}-{_max}" },
                "book": { self._sanitize_id(): name.split('_')[1] }
                }

    def internal_serialize(self):
        return len(self.get_filename()).to_bytes(I_SIZE, byteorder=BYTE_ORDER) + str.encode(self.get_filename())

    def internal_load(cls, f):
        blen = f.read(I_SIZE)
        bstr = f.read(blen)
        return blen + bstr, bstr.decode()

    @classmethod
    def create_node_from_DB(cls, db_manager, _id, hash_algorithm):
        raise NodeUnsupportedAlgorithm # algorithm not supported

    @classmethod
    def internal_data_needs_DB(cls) -> bool:
        return False # we have some data necessary to retrieve from the DB
                    # to load a WinModuleHashNode from an Apotheosis file

    def is_equal(self, other):
        if type(self) != type(other):
            return False
        try:
            same_file = self._filename == other._filename
            if not same_file:
                return False
            if type(self._hash_algorithm) != type(other._hash_algorithm):
                return False
            equal = self._id == other._id
            if not equal:
                return False
            
            return True
        except:
            return False

