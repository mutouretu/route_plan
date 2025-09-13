# methods/examples.py
# Placeholder routing methods for the framework (replace with your algorithms).
import numpy as np

def _nn_tour(D: np.ndarray):
    n = D.shape[0]
    unvisited = set(range(1, n))
    tour = [0]
    cur = 0
    while unvisited:
        nxt = min(unvisited, key=lambda j: D[cur, j])
        tour.append(nxt)
        unvisited.remove(nxt)
        cur = nxt
    tour.append(0)
    return tour

def method1(D: np.ndarray):
    """Method1: Nearest Neighbor"""
    return _nn_tour(D)

def method2(D: np.ndarray):
    """Method2: Random permutation (baseline stub)"""
    n = D.shape[0]
    rng = np.random.default_rng(1)
    perm = rng.permutation(np.arange(1, n)).tolist()
    return [0] + perm + [0]

def method3(D: np.ndarray):
    """Method3: Farthest insertion (quick heuristic)"""
    n = D.shape[0]
    far = int(np.argmax(D[0,1:])) + 1
    tour = [0, far, 0]
    remaining = [j for j in range(1, n) if j != far]
    while remaining:
        pick = max(remaining, key=lambda j: min(D[j, k] for k in set(tour)))
        best_pos, best_inc = None, float('inf')
        for i in range(len(tour)-1):
            a, b = tour[i], tour[i+1]
            inc = D[a, pick] + D[pick, b] - D[a, b]
            if inc < best_inc:
                best_inc, best_pos = inc, i+1
        tour.insert(best_pos, pick)
        remaining.remove(pick)
    return tour

def method4(D: np.ndarray):
    """Method4: 2-opt on top of NN"""
    tour = _nn_tour(D)
    improved = True
    while improved:
        improved = False
        for i in range(1, len(tour)-2):
            for j in range(i+1, len(tour)-1):
                a, b = tour[i-1], tour[i]
                c, d = tour[j], tour[j+1]
                delta = D[a, c] + D[b, d] - (D[a, b] + D[c, d])
                if delta < -1e-9:
                    tour[i:j+1] = reversed(tour[i:j+1])
                    improved = True
    return tour
