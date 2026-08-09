"""Random search baseline."""

import random

def random_allocation(budget, categories, iterations=1000):
    keys = list(categories)
    best = None
    for _ in range(iterations):
        weights = [random.random() for _ in keys]
        s = sum(weights) or 1.0
        alloc = {k: budget * (w / s) for k, w in zip(keys, weights)}
        best = alloc
    return best
