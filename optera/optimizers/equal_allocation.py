"""Equal allocation baseline."""

def equal_allocation(budget, categories):
    n = max(1, len(categories))
    per = budget / n
    return {k: per for k in categories}
