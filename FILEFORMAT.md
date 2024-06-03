# APOT File Format

This is a description of the binary file format of Apotheosis data structure. Each version of the file format is described using the [W3C Extensible Markup Language (XML) 1.0 (Fifth Edition)](https://www.w3.org/TR/xml/#sec-notation).

## Version 1

```
<apot-file>     ::= <header> <HNSW-cfg> <ep> <nodes> <EOF>
<header>        ::= <magic> <VERSION> <CRC32_CFG> <CRC32_EP> <CRC32_NODES>
<HNSW-cfg>      ::= <M> <Mmax> <Mmax0> <ef> <mL> <distance-algorithm>
                    <heuristic> <extend_candidates> <keep_pruned_conns>
                    <beer_factor>
<ep>            ::= <N_LAYER> <node>
<nodes>         ::= ( <node> )*

/* Node definition */
<node>          ::= <page-id> <neighborhoods>
<neighborhoods> ::= <N_HOODS> ( <neighborhood> )*
<neighborhood>  ::= <N_LAYER> <N_NEIGS> ( <neighbor> )*
<neighbor>      ::= <page-id>

/* We adhere to Python3 struct datatype definitions */
/* See common/constants.py for details */
<page-id>       ::= <integer> /* unsigned int (I_SIZE) */
<N_HOODS>       ::= <integer> /* unsigned int (I_SIZE) */
<N_LAYER>       ::= <integer> /* unsigned int (I_SIZE) */
<N_NEIGS>       ::= <integer> /* unsigned int (I_SIZE) */

/* Header data */
<magic>         ::= <llong>   /* long long (D_SIZE) */
<VERSION>       ::= <byte>    /* byte (C_SIZE) */
<CRC32_CFG>     ::= <integer> /* unsigned int (I_SIZE), it will be crc32(<HNSW-cfg>) */
<CRC32_EP>      ::= <integer> /* unsigned int (I_SIZE), it will be crc32(<ep>) */
<CRC32_NODES>   ::= <integer> /* unsigned int (I_SIZE), it will be crc32(<nodes>) */

/* HNSW configuration parameters */
<M>             ::= <integer> /* unsigned int (I_SIZE) */
<Mmax>          ::= <integer> /* unsigned int (I_SIZE) */
<Mmax0>         ::= <integer> /* unsigned int (I_SIZE) */
<ef>            ::= <integer> /* unsigned int (I_SIZE) */
<mL>            ::= <real>    /* double (D_SIZE) */
<heuristic>     ::= <byte>    /* byte (C_SIZE) */
<extend_candidates>     ::= <byte>    /* byte (C_SIZE) */
<keep_pruned_conns>     ::= <byte>    /* byte (C_SIZE) */
<beer_factor>           ::= <double>  /* byte (D_SIZE) */ 
<distance-algorithm>    ::= <TLSH> | <SSDEEP> /* byte (C_SIZE) */
<TLSH>          ::= #x0 /* value 0 */
<SSDEEP>        ::= #x1 /* value 1 */

<EOF>           ::= #x0 /* value 0 */
```
