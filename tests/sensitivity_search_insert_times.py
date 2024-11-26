import os
import statistics
import sys
from subprocess import Popen, PIPE
import common.utilities as util

def run_search_insert_test(M: int=4, ef: int=4, Mmax: int=16,\
                            Mmax0: int=16, algorithm="", bf: float=0.0,\
                            search_recall: int=4, dump_filename: str=None,\
                            npages: int=200, nsearch_pages: int=0):
    cmd = ["python3", "-m", "tests.search_insert_times_test"]

    cmd.extend(["--M", str(M)]);
    cmd.extend(["--ef", str(ef)]);
    cmd.extend(["--Mmax", str(Mmax)]);
    cmd.extend(["--Mmax0",  str(Mmax0)]);
    cmd.extend(["-algorithm",  str(algorithm)]);
    if bf > 0:
        cmd.extend(["--beer-factor", str(bf)]);
    cmd.extend(["--search-recall", str(search_recall)]);
    if dump_filename:
        cmd.extend(["--dump-file", str(dump_filename)]);
    cmd.extend(["--npages", str(npages)]);
    cmd.extend(["--nsearch-pages", str(nsearch_pages)]);

    process = Popen(cmd, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    return stdout, stderr

if __name__ == '__main__':
    parser  = util.configure_argparse()
    parser.add_argument('-dump', '--dump-file', type=str, help="Filename to dump Apotheosis data structure")
    parser.add_argument('-outdir', '--output-log-directory', type=str, help="Output log directory", default="logs")
    parser.add_argument('-recall', '--search-recall', type=int, default=4, help="Search recall (default=4)")
    parser.add_argument('--npages', type=int, default=1000, help="Number of pages to test (default=1000)")
    parser.add_argument('--nsearch-pages', type=int, default=0, help="Number of pages to search (default=0)")
    parser.add_argument('--factor', type=int, default=10, help="Max values of M, Mmax, and Mmax0 (default=10)")
    parser.add_argument('--extra', action='store_true', help="Enable extra functionality")
    parser.add_argument('--extra2', action='store_true', help="Enable extra functionality")
    args    = parser.parse_args()
    util.configure_logging(args.loglevel.upper())
   
    filename = f"log_{args.factor}_{args.npages}_{args.nsearch_pages}.out"

    if args.factor == 0:
        EF      = [4]
        M       = [4]
        Mmax    = [16]
        Mmax0   = [16]
        if args.extra:
            EF      = [8]
            M       = [16]
            Mmax    = [16]
            Mmax0   = [32]
            filename = f"log_{args.factor}{args.factor}_{args.npages}_{args.nsearch_pages}.out"
        if args.extra2:
            EF      = [8]
            M       = [8]
            Mmax    = [16]
            Mmax0   = [16]
            filename = f"log_{args.factor}{args.factor}0_{args.npages}_{args.nsearch_pages}.out"
    else:
        EF      = range(4, 2*(args.factor + 1), 2)
        M       = range(4, 4*(args.factor + 1), 4)
        Mmax    = range(16, 16*(args.factor + 1), 4)
        Mmax0   = range(16, 16*(args.factor + 1), 4)
    
    
    equal_hashes = set()

    f = open(os.path.join(args.output_log_directory, filename), "w")
    f.write(f'TYPE,EF,M,MMAX,MMAX0,TIME\n')
   
    for ef in EF:
        for m in M:
            for mmax in Mmax:
                insert_list     = []
                search_exact    = []
                search_approx   = []
                for mmax0 in Mmax0:
                    stdout, stderr = run_search_insert_test(m, ef, mmax, mmax0,\
                            args.distance_algorithm, args.beer_factor,\
                            args.search_recall, args.dump_file, npages=args.npages, nsearch_pages=args.nsearch_pages)
                    # get search and insert times
                    stdout_lines = [s.decode("utf-8") for s in stdout.splitlines()]
                    for line in stdout_lines:
                        if "SEARCH" not in line:
                            if "INSERT" not in line:
                                continue
                            else:
                                insert_time = float(line.split(':')[1].replace('ms', ''))
                                
                                f.write(f'I,{ef},{m},{mmax},{mmax0},{insert_time}\n')
                                insert_list.append(insert_time)
                        else:
                            search_time = float(line.split(':')[1].replace('ms', ''))
                            if "SEARCH EXACT" in line:
                                search_method = "SE"
                                search_exact.append(search_time)
                            else:
                                search_method = "SA"
                                search_approx.append(search_time)
                            f.write(f'{search_method},{ef},{m},{mmax},{mmax0},{search_time}\n')
                # get equal hashes
                stderr_lines = [s.decode("utf-8") for s in stderr.splitlines()]
                for line in stderr_lines:
                    try:
                        if "COLLISION" in line:
                            print(f"Collision found: {line}")
                        else:
                            line  = line.split("\"")
                            _hash = line[1]
                            equal_hashes.add(_hash)
                    except Exception as e:
                        print(f"Exception occurred with line {line}")
                        print(e)
    f.close()
    
    f = open(f"equal_hashes_{args.factor}_{args.npages}.out", "w")
    for equal_hash in equal_hashes:
        f.write(equal_hash)
        f.write("\n")
    f.close()
    print("[+] Done!")                    
