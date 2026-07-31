# Bundles

Infer bundles live here — the self-contained pipeline directories
`chachak.bundle_export` writes (`pipeline.json`, `models/`, a vendored
`runtime/`, and an `infer.py` that runs them with no chachak checkout).

Anything with a `pipeline.json` in it is offered as a model source by the video
"Run model inference…" action and by camera live inference, so **copying a
bundle directory in here is all it takes to be able to run it** — the bundle's
manifest supplies the model, the person detector and the pipeline geometry.

Subdirectories are scanned too, so bundles can be grouped however you like.
Change the location in Training settings (`bundles_root`).

Two things worth knowing:

- A bundle's own export drops one beside every artifact under `exports_root`
  (`<name>-bundle/`); those are found via *that* setting, not this directory.
  Copy or move one here if you'd rather keep the two apart.
- A `.engine` is a TensorRT plan tied to the GPU model and TensorRT version it
  was built on. Press "Sync bundle" with the load test enabled after copying one
  in from another machine — that is the only check that proves it loads here.
