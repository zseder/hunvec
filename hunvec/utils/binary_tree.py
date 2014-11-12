"""
Copied from http://rosettacode.org/wiki/Huffman_coding#Python and modified.
"""
#import logging

from heapq import heappush, heappop, heapify
import numpy


class BinaryTreeEncoder():
    def __init__(self, symb2freq):
        self.create_binary_tree(symb2freq)
        self.index_prefixes()

    def create_binary_tree(self, symb2freq):
        """Huffman encode the given dict that maps symbols to weights"""
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
        self.tree = dict(heappop(heap)[1:])

    def index_prefixes(self):
        prefixes = set()
        for w, code in self.tree.iteritems():
            for i in xrange(len(code)):
                prefixes.add(code[:i])

        alpha_sorted = sorted(prefixes)
        len_sorted = sorted(alpha_sorted, key=lambda x: len(x))
        self.prefix_index = dict((prefix, i) for i, prefix in
                                 enumerate(len_sorted))

    def word_encoder(self, word):
        return self.to_long(self.tree[word])

    def to_long(self, code):
        new_code = numpy.ones(len(self.prefix_index))
        new_code *= -1
        for pr_i in xrange(len(code)):
            pr = code[:pr_i]
            if code[pr_i] == '0':
                new_code[self.prefix_index[pr]] = 0
            else:
                new_code[self.prefix_index[pr]] = 1
        return new_code

    def to_short(self, code):
        return filter(lambda x: x != -1, code)


if __name__ == '__main__':
    import sys
    d = dict(line.split() for line in open(sys.argv[1]))
    e = BinaryTreeEncoder(d)
    print e.word_encoder('a')
