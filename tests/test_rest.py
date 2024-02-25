import requests
import base64
import time

SLEEP_TIME = 20 # in seconds

def test(_hash: str):
    print(f"[*] Asking for \"{_hash.decode('utf-8')}\" ...")
    base_url = "http://localhost:5000"
    search_url = f"{base_url}/search/knn/5/tlsh/{base64.b64encode(_hash).decode('utf-8')}"
    response = requests.get(search_url)

    while(response.status_code == 202):
        print(f"[+] Response ({response.status_code}): \"{response.text}\"")
        print(f"[+] Results are not ready, waiting {SLEEP_TIME} seconds to try again ...")
        time.sleep(SLEEP_TIME)
        response = requests.get(response.url)

    if response.status_code == 200:
        print(f"[+] Response 200 obtained!")
        results = base64.b64decode(response.text).decode('utf8')
        print(results)
    else:
        print(f"[-] Response ({response.status_code}): \"{response.text}\"")

if __name__ == "__main__":
    hash1 = b"T12B81E2134758C0E3CA097B381202C62AC793B46686CD9E2E8F9190EC89C537B5E7AF4C"
    hash2 = b"T10381E956C26225F2DAD9D5C2C5C1A337FAF3708A25012B8A1EACDAC00B37D557E0E714"
    hash3 = b"T173819073B82798B7CE3550E6211D65B718E84E87F534ACF46D6AF51FE27A1E020D1B08"
    hash4 = b"T1DF8174A9C2A506F9C6FFC292D6816333FEF1B845C419121A0F91CF5359B5B21FA3A304"
    hash5 = b"T1DF8174A9C2A506F9C6FFC292D6816333FEF1B845C419121A0F91CF5359B5B21FA3A305"

    hash_nodes = [hash1, hash2, hash3, hash4, hash5]
    for _hash in hash_nodes:
        test(_hash)
