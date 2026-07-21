"""Helpers for combined multi-model evaluation (evaluate 2+ models as one).

See ``TrainedModelAdmin.evaluate``: selecting 2+ trained models merges their
predictions into a single eval row. Combining is only valid for
*complementary* detectors — two models sharing a class name would double-count
it in the merged result — so callers must check :func:`overlapping_class_names`
before combining and refuse when it's non-empty.
"""

from collections import Counter


def overlapping_class_names(models) -> set[str]:
    """Class names shared by 2+ of the given models' ``TrainedModel.classes``.

    Empty when every model's class list is disjoint from every other's — the
    only case where merging their predictions is well-defined.
    """
    counts = Counter()
    for model in models:
        for name in set(model.classes or []):
            counts[name] += 1
    return {name for name, count in counts.items() if count > 1}
