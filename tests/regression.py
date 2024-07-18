from apotheosis_winmodule import ApotheosisWinModule
import common.utilities as utils

from pytest_regressions.data_regression import DataRegressionFixture
from datalayer.hash_algorithm.tlsh_algorithm import TLSHHashAlgorithm
from datalayer.node.hash_node import HashNode
import base64
import requests
import json

APOTHEOSIS_HOST = "localhost:5000"
HASHES = [
    "T1A9817D1B87CF1EFCC4061974064FDC192098C0A046E496970F63B06779DE292F538F44",
    "T1F981478326FD18C5F5F3AFB45FF98024D832BD921A6AC56E01054A4E9AB2E54CD35B32",
    "T143816C05ABE51205E167F7702FF6C0AB892F7855CA3E4B6F004887671B53A407A16F7D",
    "T12481AB5067EE790DF1B2BD709E71C4354836FE105D73831F3680A989A5B4D7AC936B22",
    "T12781F94391825F638AC5A520482D324D792DE7BA035B39C392E89BF0964EFD67B302CD",
    "T1BA81F997BE82F4E3E7A8A840451E53473D2E823466320ADB76E3DF862456AC3273115B",
    "T13A819352F3115536C5E011507A4E76EA773ED8B14B5E5EEB9250ECEE8008BE73B7022B",
    "T12F81C557F34428230491902182CDAEEF755DB0F01B9AA8C7B382DCB97D1DAAAB674701",
    "T16681941BE2416823BA509064F71CB5EB7F0C50321B6A38CBF313AE95155D9E3A23625A",
    "T18281C79196908553B09590E09E8E7FF5FA0FE1380B4554CBB3D038FE6118EE3BB241AB"
]

def create_dict_result(found, node, result_dict):

    result = {"found": found,\
                "query": node.as_dict(),\
                "hashes":
                    {key: value[0].as_dict() for key, value in result_dict.items()}
                }

    print(f"Veamos: {result_dict.items()}")
    return result

def test_insertion(data_regression: DataRegressionFixture):
    apo_model = ApotheosisWinModule(M=64, ef=32, Mmax=64, Mmax0=128,
                           distance_algorithm=TLSHHashAlgorithm)
        
    print("[*] Building ApotheosisWinModule with TLSH ...")
    for hash in HASHES:
        apo_model.insert(HashNode(hash, TLSHHashAlgorithm))
    
    found, exact, result_dict = apo_model.knn_search(query=HashNode(HASHES[3], TLSHHashAlgorithm), k=1, ef=4)
    result_dict = create_dict_result(found, exact, result_dict)

    

    data_regression.check(result_dict)

def test_search(data_regression: DataRegressionFixture):
    apo_model = ApotheosisWinModule(M=64, ef=32, Mmax=64, Mmax0=128,
                           distance_algorithm=TLSHHashAlgorithm)
    
    for hash in HASHES:
        apo_model.insert(HashNode(hash, TLSHHashAlgorithm))
    
    search_results = []
    for hash in HASHES:
        found, exact, result_dict = apo_model.knn_search(HashNode(hash, TLSHHashAlgorithm), k=1, ef=4)
        result_dict = create_dict_result(found, exact, result_dict)
        search_results.append(result_dict)
    
    data_regression.check(search_results)

def test_search_threshold(data_regression: DataRegressionFixture):
    apo_model = ApotheosisWinModule(M=64, ef=32, Mmax=64, Mmax0=128,
                           distance_algorithm=TLSHHashAlgorithm)
    
    for hash in HASHES:
        apo_model.insert(HashNode(hash, TLSHHashAlgorithm))
    
    search_results = []
    for hash in HASHES:
        found, exact, result_dict = apo_model.threshold_search(HashNode(hash, TLSHHashAlgorithm), threshold=100, n_hops=4)
        result_dict = create_dict_result(found, exact, result_dict)
        search_results.append(result_dict)
    
    data_regression.check(search_results)


def test_deletion(data_regression: DataRegressionFixture):
    apo_model = ApotheosisWinModule(M=64, ef=32, Mmax=64, Mmax0=128,
                           distance_algorithm=TLSHHashAlgorithm)
    
    for hash in HASHES:
        apo_model.insert(HashNode(hash, TLSHHashAlgorithm))
    
    for hash in HASHES[:5]:
        apo_model.delete(HashNode(hash, TLSHHashAlgorithm))
    
    remaining_hashes = HASHES[5:]
    search_results = []
    for hash in remaining_hashes:
        found, exact, result_dict = apo_model.knn_search(HashNode(hash, TLSHHashAlgorithm), 1)
        result_dict = create_dict_result(found, exact, result_dict)
        search_results.append(result_dict)
    
    data_regression.check(search_results)
