from db_manager import DBManager
import tlsh
import ssdeep
import numpy as np
import time

class Bruteforce:
    def __init__(self):
        self.list_pages = self.get_db_pages()[:75000]

    def measure_precission(self):
        init_time = time.time()
        similarities = []
        for i, _ in enumerate(self.list_pages):
            print(f"Comparing node {i} with each one...")
            similarities.append([])
            for j, _ in enumerate(self.list_pages):
                similarities[i].append((ssdeep.compare(self.list_pages[i].hashSSDEEP, self.list_pages[j].hashSSDEEP) - 100) * -1)
            similarities.append(similarities[i])
        end_time = time.time() - init_time
        print(f"Elapsed time: {end_time}")
        np.savetxt("bruteforce.txt", similarities)
        print("Elapsed time " + end_time)
        return similarities

    def get_db_pages(self):
        dbManager = DBManager()
        print("Getting pages from DB...")
        return dbManager.get_all_pages()
    
bf = Bruteforce()
bf.measure_precission()
