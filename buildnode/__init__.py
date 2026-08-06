"""buildnode — a slim TensorRT engine/bundle builder that runs on someone else's GPU.

A TensorRT engine only deserializes on the GPU model and TensorRT version that
built it, so an engine built on the box that trains is useless anywhere else.
This package is the other end of that problem: a small HTTP service you deploy on
each GPU machine, which accepts an already-exported ``.onnx`` plus its
``.meta.json``, compiles the engine on *its* GPU, assembles a chachak infer bundle
around it, and hands the bundle back as a tarball.

It deliberately does NOT carry the training stack. No torch, no dataloaders, no
architectures — a build node cannot train, cannot evaluate, and cannot accept a
``.pt``. Exporting a checkpoint to ONNX is CPU work that the trainer already does
safely while the GPU is busy (``friendy_chachkalica/service.py``'s
``/export_onnx``), so it stays there and only the GPU-bound half travels.

The counterpart on the Django side is ``chachkalica/training/services/buildnode.py``
and the ``BuildNode`` admin. See ``chachkalica/docs/build-nodes.md``.
"""

NODE_VERSION = "1"
