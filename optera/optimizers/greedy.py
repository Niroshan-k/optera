"""Greedy allocation baseline."""

def greedy_allocation(budget, categories):
    """Very simple greedy baseline: allocate proportional to categories weights if provided."""
    total = sum(categories.values()) if categories else 1
    return {k: budget * (v / total) for k, v in (categories or {}).items()}
