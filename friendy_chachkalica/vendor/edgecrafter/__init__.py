"""Minimal vendored EdgeCrafter/ECDet runtime.

Upstream: https://github.com/Intellindust-AI-Lab/EdgeCrafter
Commit:   706d037c17c1703bb97f42a35d269959b511b5be (2026-07-23)
License:  Apache-2.0 (see LICENSE in this directory)
Paper:    "EdgeCrafter: Compact ViTs for Edge Dense Prediction via
          Task-Specialized Distillation" (arXiv:2603.18739)

Upstream ships no ``setup.py``/``pyproject.toml``, so there is nothing to pip
install; the model code is vendored here the same way ``vendor/yolox`` is.

What was taken, verbatim, from upstream ``ecdetseg/``:

    core/workspace.py       engine/core/workspace.py    (@register / create)
    core/yaml_utils.py      engine/core/yaml_utils.py   (load/merge, __include__)
    edgecrafter/*           engine/edgecrafter/*         (the model)
    configs/ecdet*.yml      configs/ecdet/*.yml          (the four variants)

What was pruned or replaced, and why:

    engine/core/_config.py      dropped. Only ``YAMLConfig`` used it, and it
    engine/core/yaml_config.py   imports SummaryWriter/GradScaler/DataLoader.
                                 ``YAMLConfig.model`` is just
                                 ``create(cfg['model'], merge_config(cfg))``,
                                 which ml/adapters/ecdet.py does directly.
    engine/misc/dist_utils.py   replaced by misc/dist_utils.py here: upstream's
                                 is 268 lines of DDP/FSDP wrapping that begins
                                 ``from ..data import DataLoader``, and only two
                                 of its functions are ever called.
    engine/{data,solver,optim}  dropped. We bring our own dataset, augmentation
    tools/                       loop, optimizer and exporters.
    configs/dataset/coco.yml    replaced by configs/_friendy_dataset.yml, which
                                 carries the class count and nothing else, so no
                                 COCO path is ever resolved.

Two upstream behaviours the caller must respect (ml/adapters/ecdet.py does):

1. ``load_config(file_path, cfg=dict())`` has a **mutable default argument**, so
   consecutive calls accumulate into one shared dict. Always pass ``cfg={}``.
2. ``create()`` mutates the schema dicts inside ``GLOBAL_CONFIG`` while it
   resolves ``__inject__`` entries, and ``merge_config`` copies those dicts in by
   reference. Each call restores the registered defaults first, so sequential
   builds are fine, but building two models **concurrently** in one process is
   not. Build one adapter at a time.

Also note ``register()`` asserts a name is not already registered, so this
package must be imported under exactly one module spelling per process — see the
single-spelling import in ``ml/adapters/ecdet.py``.
"""
