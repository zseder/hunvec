def viterbi(start, A, end, probs, n_classes):
    V = [{}]
    path = {}
    states = range(n_classes)

    # Initialize base cases (t == 0)
    for y in states:
        V[0][y] = start[y] * probs[0][y]
        path[y] = [y]

    # Run Viterbi for t > 0
    for t in range(1, probs.shape[0]):
        V.append({})
        newpath = {}

        for y in states:
            if t != probs.shape[0] - 1:
                (prob, state) = max(
                    (V[t-1][y0] * A[y0][y] * probs[t][y], y0)
                    for y0 in states)
            else:
                (prob, state) = max(
                    (V[t-1][y0] * A[y0][y] * probs[t][y] * end[y], y0)
                    for y0 in states)
            V[t][y] = prob
            newpath[y] = path[state] + [y]

        # Don't need to remember the old paths
        path = newpath

    n = 0
    if probs.shape[0] != 1:
        n = t
    (prob, state) = max((V[n][y], y) for y in states)
    return (prob, path[state])
