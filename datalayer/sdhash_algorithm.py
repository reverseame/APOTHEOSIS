import subprocess
from hash_algorithm import HashAlgorithm
import time
import execnet

class SDHashAlgorithm(HashAlgorithm):
    def compare(hash1, hash2):
        time_init = time.time()
        #TODO: change this, avoid constant paths
        with open('/home/dhuici/ramdisk/hashes.sdbf', 'w') as hashes_file:
            hashes_file.write(hash1 + "\n" + hash2)
        time_createfile = time.time() - time_init
        time_init = time.time()
        result = subprocess.run('sdhash -c /home/dhuici/ramdisk/hashes.sdbf', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        time_run = time.time() - time_init
        print("go python2")
        gateway = execnet.makegateway("popen//python=python2")
        # Código de Python 2 que deseas ejecutar
        codigo_python2 = """
        def funcion_python2():
            print("Llamada a función en Python 2")
        funcion_python2()
        """
        print("test")
        # Ejecuta el código en el gateway de Python 2
        channel = gateway.remote_exec(codigo_python2)
        
        # Obtiene la salida del canal
        salida_python2 = channel.receive()

        print(f"Create file: {time_createfile} / run: {time_run}")
        if result.stdout == "": # No match at all
            return 0
        else:
            print(result.stdout)
            return int(result.stdout[2:]) # Remove the first || chars
        

