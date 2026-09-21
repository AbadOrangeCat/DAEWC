"""Numbers shared by manuscript prose and tables, computed before rounding."""
from math import isfinite
import numpy as np


def primary_source_changes(runs, seeds=(42, 43, 44)):
    """Return the K=80 DAEWC means, rejecting incomplete or repeated evidence."""
    result = {}
    for domain, name in [("health", "ReferenceHealthSourceChange"),
                         ("public_affairs", "ReferencePublicSourceChange"),
                         ("entertainment", "ReferenceEntertainmentSourceChange")]:
        group = sorted((r for r in runs if r["method"] == "daewc"
                        and r["shots_per_class"] == 80 and r["domain"] == domain),
                       key=lambda r: r["seed"])
        if [r["seed"] for r in group] != sorted(seeds):
            raise ValueError(f"{domain}: expected exactly one result per seed {seeds}")
        values = [r["delta_source_pp"] for r in group]
        if not all(isfinite(value) for value in values):
            raise ValueError(f"{domain}: non-finite source change")
        result[name] = float(np.mean(values))
    return result


def source_change_macros(runs):
    values = primary_source_changes(runs)
    return "".join(f"\\newcommand{{\\{name}}}{{{value:.2f}}}\n" for name, value in values.items())
