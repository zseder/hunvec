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
    return sorted(heappop(heap)[1:], key=lambda p: (len(p[-1]), p))


def vector_encoder(t):
    prefixes = reduce(lambda x, y: x | y,
                      [set(code[:i] for i in
                           xrange(len(code))) for w, code in t])
    alpha_sorted = sorted(prefixes)
    len_sorted = sorted(alpha_sorted, key=lambda x: len(x))
    prefixes = dict((prefix, i) for i, prefix in enumerate(len_sorted))
    encoder = {}
    for w, code in t:
        new_code = numpy.ones(len(prefixes))
        new_code *= -1
        for pr_i in xrange(len(code)):
            pr = code[:pr_i]
            if code[pr_i] == '0':
                new_code[prefixes[pr]] = 0
            else:
                new_code[prefixes[pr]] = 1
        encoder[w] = new_code
    return encoder
