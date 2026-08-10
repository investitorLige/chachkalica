"""The two distributed helpers the vendored model actually calls.

Upstream's ``engine/misc/dist_utils.py`` is a 268-line DDP/FSDP/torch.compile
wrapper for its own solver, and it starts with ``from ..data import DataLoader``
— which would drag the pruned ``engine/data`` package (pycocotools, the COCO
dataset, the augmentation pipeline) back in. Only these are referenced:

    edgecrafter/criterion.py:25  get_world_size, is_dist_available_and_initialized
    edgecrafter/ecvit.py:460     is_dist_available_and_initialized

Both are used to average a box count across ranks and to barrier around the
backbone-weight download. The trainer runs single-process, so both answer
"not distributed" — but they are kept faithful rather than stubbed so a future
DDP trainer keeps working. ``get_rank``/``is_main_process`` round out the set at
no cost, matching upstream's names and semantics.
"""

import torch
import torch.distributed


def is_dist_available_and_initialized() -> bool:
    if not torch.distributed.is_available():
        return False
    if not torch.distributed.is_initialized():
        return False
    return True


def get_rank() -> int:
    if not is_dist_available_and_initialized():
        return 0
    return torch.distributed.get_rank()


def get_world_size() -> int:
    if not is_dist_available_and_initialized():
        return 1
    return torch.distributed.get_world_size()


def is_main_process() -> bool:
    return get_rank() == 0
