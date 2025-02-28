import unittest
from apotheosis import Apotheosis
from datalayer.hash_algorithm.tlsh_algorithm import TLSHHashAlgorithm
from datalayer.node.hash_node import HashNode
from datalayer.node.winpage_hash_node import WinPageHashNode
from unittest.mock import patch
from datalayer.db_manager import DBManager




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

APROX_HASHES = ["T12B81E2134758C0E3CA097B381202C62AC793B46686CD9E2E8F9190EC89C537B5E7AF4C",
                "T1DF8174A9C2A506F9C6FFC292D6816333FEF1B845C419121A0F91CF5359B5B21FA3A304"]



class TestApotheosis(unittest.TestCase):

    def setUp(self):
        self.apo_model = Apotheosis(M=64, ef=32, Mmax=64, Mmax0=128, distance_algorithm=TLSHHashAlgorithm)
        for hash in HASHES:
            self.apo_model.insert(HashNode(hash, TLSHHashAlgorithm))

    def test_insertion(self):
        found, exact, _ = self.apo_model.knn_search(query=HashNode(HASHES[3], TLSHHashAlgorithm), k=1, ef=4)

        self.assertTrue(found)
        self.assertEqual(exact.get_id(), "T12481AB5067EE790DF1B2BD709E71C4354836FE105D73831F3680A989A5B4D7AC936B22")

    def test_search_exact(self):
        expected_founds = [True for i in range (0, 10)]
        actual_founds = []
        actual_hashes = []
        for hash in HASHES:
            found, exact, result_dict = self.apo_model.knn_search(HashNode(hash, TLSHHashAlgorithm), k=1, ef=4)
            actual_hashes.append(exact.get_id())
            actual_founds.append(found)

        self.assertEqual(actual_founds, expected_founds)
        self.assertEqual(actual_hashes, HASHES)

    def test_search_approximate(self):
        expected_founds = [False, False]
        expected_distances = [236, 224]
        actual_distances = []
        actual_founds = []
        for hash in APROX_HASHES:
            found, _, result_dict = self.apo_model.knn_search(HashNode(hash, TLSHHashAlgorithm), k=1, ef=4)
            actual_distances.append(list(result_dict.keys())[0])
            actual_founds.append(found)

        self.assertEqual(actual_founds, expected_founds)
        self.assertEqual(actual_distances, expected_distances)

    def test_deletion(self):
        for hash in HASHES[:5]:
            self.apo_model.delete(HashNode(hash, TLSHHashAlgorithm))

        expected_founds = [False, False, False, False, False, True, True, True, True, True]
        actual_founds = []
        for hash in HASHES:
            found, _, _ = self.apo_model.knn_search(HashNode(hash, TLSHHashAlgorithm), 1)
            actual_founds.append(found)

        self.assertEqual(actual_founds, expected_founds)


    def mock_get_winmodule_data_by_hash(self, algorithm, hash_value):
        """Mock function for get_winmodule_data_by_hash"""
        return WinPageHashNode(hash_value, algorithm)

    def test_dump_load(self):
        self.apo_model = Apotheosis(
            M=4, ef=4, Mmax=8, Mmax0=16,
            heuristic=False, extend_candidates=False, 
            keep_pruned_conns=False, beer_factor=False,
            distance_algorithm=TLSHHashAlgorithm
        )

        hash1 = "T10381E956C26225F2DAD9D5C2C5C1A337FAF3708A25012B8A1EACDAC00B37D557E0E714" 
        hash2 = "T1458197A3C292D1EC8566C6A2C6516377FA743E0F8120BA49CFD1CF812B66B60D75E316" 

        node1 = WinPageHashNode(hash1, TLSHHashAlgorithm)
        node2 = WinPageHashNode(hash2, TLSHHashAlgorithm)

        # Using self to reference the mock function
        with patch.object(DBManager, 'connect', return_value=None), \
            patch.object(DBManager, 'get_winmodule_data_by_hash', side_effect=self.mock_get_winmodule_data_by_hash) as mock_method:
            self.apo_model.insert(node1)
            self.apo_model.insert(node2)

            self.apo_model.dump("TestApo", compress=False)
            self.apo_model.load('TestApo', TLSHHashAlgorithm, WinPageHashNode)

            _, exact1, _ = self.apo_model.knn_search(HashNode(hash1, TLSHHashAlgorithm), k=1, ef=4)
            _, exact2, _ = self.apo_model.knn_search(HashNode(hash2, TLSHHashAlgorithm), k=1, ef=4)

            self.assertEqual(exact1.get_id(), hash1)
            self.assertEqual(exact2.get_id(), hash2)


if __name__ == '__main__':
    unittest.main()
