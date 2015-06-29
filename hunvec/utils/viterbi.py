import numpy
import theano


def viterbi(start, A, end, tagger_out, n_classes, return_probs=False):
    V = numpy.zeros(tagger_out.shape, dtype=theano.config.floatX)
    path = {}
    states = range(n_classes)

    # Initialize base cases (t == 0)
    V[0] = start + tagger_out[0]
    for y in states:
        path[y] = [y]

    # Run Viterbi for t > 0
    for t in range(1, tagger_out.shape[0]):
        newpath = {}

        l_probs = V[t-1] + A.T + tagger_out[t].reshape((n_classes, 1))
        if t == tagger_out.shape[0] - 1:
            l_probs += end.reshape((n_classes, 1))
        l_states = l_probs.argmax(axis=1)
        probs = l_probs[states, l_states]
        V[t] = probs
        for y in states:
            newpath[y] = path[l_states[y]] + [y]

        # Don't need to remember the old paths
        path = newpath

    n = 0
    if tagger_out.shape[0] != 1:
        n = t
    if return_probs:
        return V

    state = V[n].argmax()
    prob = V[n][state]
    return (prob, path[state])
