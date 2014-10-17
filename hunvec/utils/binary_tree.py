"""
Copied from http://rosettacode.org/wiki/Huffman_coding#Python and modified.
"""
from heapq import heappush, heappop, heapify

import numpy


def create_binary_tree(symb2freq):
    """Huffman encode the given dict mapping symbols to weights"""
    heap = [[wt, [sym, ""]] for sym, wt in symb2freq.items()]
    heapify(heap)
    while len(heap) > 1:
        lo = heappop(heap)
        hi = heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    words = sorted(heappop(heap)[1:], key=lambda p: (len(p[-1]), p))
    btree_len = max(len(b) for _, b in words)
    d = {}
    for w, b in words:
        d[w] = numpy.zeros(btree_len, dtype='int8')
        for ch_i in xrange(len(b)):
            if b[ch_i] == '1':
                d[w][ch_i] = 1
    return d
