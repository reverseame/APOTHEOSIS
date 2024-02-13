
BYTE_ORDER  = 'big' # big-endian format
VERSIONFILE = 1
EOF         = 0

TLSH        = 0
SSDEEP      = 1
SDHASH      = 2

HEADER_SIZE = 21 
CFG_SIZE    = 36

"""We use the definition of standard sizes as in
https://docs.python.org/3/library/struct.html#struct-format-strings
        = means we want standard sizes
        =I: unsigned int, 4B
        =d: double, 8B
        =c: char, 1B
        =?: bool, 1B
"""
I_SIZE = 4
D_SIZE = 8
C_SIZE = 1
