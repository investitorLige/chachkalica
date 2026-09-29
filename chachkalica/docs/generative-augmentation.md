# Generative Augmentation

Prompted image augmentation with **FireRed-Image-Edit-1.1**: take annotated
training images, ask the model to change how they *look* (night, compression,
low light, motion blur, fluorescent light) without moving anything, check that
nothing moved, and keep the source's labels on the results.

This is dataset **generation**, not a training-time transform. FireRed takes
seconds to minutes per image, so it runs ahead of training and produces an
ordinary dataset folder. The flip/crop checkboxes on an experiment still apply
on top of it at training time.

```
source dataset ──► Studio: prompt, preview on a few images, keep what works
                        │
                        ▼
                  Dataset build ──► for a fraction of the images:
                                      FireRed edit (cached)
                                      ─► geometry validation
                                      ─► accepted: image + copied label
                                      ─► every attempt: manifest line
                        │
                        ▼
      new dataset (originals + accepted variants) ──► pick it as a train dataset
```

## The section

**Generative augmentation** in the admin has three tabs:

- **Prompt library**: named, reusable prompts. Five presets are seeded
  (`night_cctv`, `compression`, `low_light`, `motion_blur`,
  `fluorescent_indoor`). Anything you write in the studio can be saved here.
- **Studio**: one row per session. A session is one dataset, its label source,
  and a handful of preview images. Add a session, and you land on its studio
  page.
- **Dataset builds**: every build, with a live report.

### The studio page

1. **Preview images.** 3 by default, picked on an even stride from the labelled
   images that have boxes. *Pick other preview images* draws a random set. Only
   prompts generated afterwards use the new set.
2. **New prompt.** Start from a preset or write one, then set seed, steps and
   CFG, plus the editor (FireRed or the mock), the weights and Lightning.
   *Generate previews* queues one job, which runs the prompt on every preview
   image. You can prompt as many times as you like; every prompt becomes its
   own card.
3. **Prompt cards.** Each one shows the generated image per preview image,
   with the **source's labels drawn on it**. Those are exactly what the build
   will copy, so they should still fit. A toggle flips the whole page between
   generated and original, with or without labels. Under each image is the
   verdict (accepted, or rejected with the reason) plus the evidence (min
   similarity, drift, how many boxes were checked).
   - **keep for build**: whether the prompt goes into the build.
   - **Re-roll seed**: the same prompt at seed + 1, as a new card.
   - **Save to library**: upserts the prompt by name.
4. **Validation.** Changing the thresholds re-checks every existing preview.
   No GPU is involved, because verdicts are recomputed from cached images.
5. **Build augmented dataset.** Set the fraction, variants per image, seed,
   per-class sampling weights (plus background), whether to include the
   originals, and whether to force regeneration. The estimate line multiplies
   it out using the preview speed measured so far.

### A build

- **Selection.** `round(fraction × labelled images)` images, drawn by weighted
  sampling without replacement (seeded). An image's weight is the highest
  class weight among the classes it contains. Background images (labelled
  empty) use the background weight, and a weight of 0 excludes an image.
  Images with **no label file are never augmented**: there is nothing to
  preserve, and copying an "empty" label onto them would turn whatever is in
  them into background.
- **Prompts per image.** `variants_per_image` distinct kept prompts (a random
  subset). When you ask for more variants than there are prompts, prompts
  repeat with seed + 1.
- **Output**, under the source root next to every other dataset:

  ```
  <name>/images/                     originals (hard-linked) + accepted variants
  <name>/labels/                     one .txt per image, same stem
  <name>/classes.txt                 copied verbatim
  <name>/genaug_manifest.jsonl       one line per attempted variant
  <name>/genaug_build.json           the whole build configuration
  ```

  Variants are named `frame_000241__firered_night_cctv__seed42.jpg`, with the
  label `frame_000241__firered_night_cctv__seed42.txt`. A Dataset row is
  registered for the output, so it appears in the experiment picker.
- **Why the originals are in there.** An experiment pairs every model with
  every train dataset as a **separate run**, so a variants-only dataset could
  not be trained on *alongside* its source. Hard links make the copy free on
  disk.
- **Manifest record:** source image and label, generated file, preset name,
  full prompt and negative prompt, seed, editor settings (model and
  quantization), steps/CFG, the full validation result, accepted, cached,
  edit time, timestamp, build id, split. Errors get a line too, with the error.
- **Caching.** Every variant is cached under `data/genaug/cache/`, keyed by
  source file (path + size + mtime), editor settings, prompt, negative prompt,
  steps, CFG and seed. Studio previews and builds share the cache, so an image
  you previewed is not paid for twice. *regenerate even if cached* is `--force`.
- **Failures** stay per variant: the error is logged to the manifest, counted,
  and the build moves on.
- **GPU busy.** When the backend refuses to load because the GPU is taken (a
  training run, say), the build **waits** and retries every minute. It shows
  the reason on the report and does not burn through the remaining images as
  errors. *Stop after the current image* ends it; everything accepted so far
  stays.

## Training on it

Add the output dataset to an experiment as a **train** dataset with label
source **source labels**. Keep your usual val/test sets.

**Leakage guard.** An experiment that trains on a build while using that
build's source (or another build of it) as val/test is refused when the config
is generated. The build contains the source's images, so that would evaluate
the model on pictures it trained on. Point a studio only at a training split.

## Validation

`genaug/services/validation.py`. The contract is
`AugmentationValidator.validate(original, generated, boxes) -> ValidationResult`
(`valid`, `bbox_drift`, `reason`, plus per-box detail), so another validator
(running the existing detector, say) plugs in without touching anything else.

The default `GeometryValidator` compares **structure, not pixels**. Every
prompt here changes pixels everywhere by design, so both images are first
turned grey, blurred and **locally normalized** (zero mean, unit variance in a
small window). That keeps edges and shapes and discards brightness, contrast,
colour and most noise. Then:

| check | how | rejects when |
|---|---|---|
| scene shift | masked correlation search over the **background** (boxes masked out) | the background moved > `max_global_shift` of the diagonal (camera moved / reframed) |
| box similarity | each box resampled to a 48-cell grid; normalized cross-correlation at the labelled position | < `min_box_similarity`: the object changed shape, vanished or was replaced |
| box drift | the same correlation searched over a ±15% margin; offset of the best match, as a fraction of the box's size | > `max_bbox_drift`, and the best match beats the labelled position by > 0.1 |

Boxes under 12 px on their short side, or with no structure at all, are
**skipped** (counted, never used to reject).

**Calibration** was done on 48 real frames from five PPE/person datasets with
simulated edits (defaults: drift 0.05, similarity 0.45, scene shift 0.02):

| edit (should pass) | accepted |  | edit (should fail) | wrongly accepted |
|---|---|---|---|---|
| night (gamma, dim, noise) | 47/48 |  | object moved 15% | 0/48 |
| JPEG quality 8 | 48/48 |  | object removed (inpainted) | 1/48 |
| 11 px motion blur | 45/48 |  | object replaced | 0/48 |
| colour cast + contrast | 48/48 |  | camera shift 3% | 0/48 |
| resample + night | 46/48 |  | object scaled 1.25× | 1/48 |
| blur + JPEG + night together | 37/48 |  | | |

Camera shifts were reported as scene shifts 48/48, and object moves as box
problems 48/48. The two that slipped through sat just above the similarity
threshold.

**What it does not catch:** a colour or identity change that keeps the
shape. A hi-vis vest recoloured to grey, for example, is invisible to a
photometric-invariant check, and that is exactly what a lighting prompt is
allowed to do. A detector-based validator is the natural second validator.
Real FireRed outputs have not been through it yet (see *Status*), so tune the
thresholds on the studio's previews. That is what they are for.

## The backend

`ml_backends/genaug/` runs as the `genaug-backend` container (port 9092), laid
out like `vlm-backend`: one editor warm, a lock serializing edits, and image
**paths** on the shared `data/` mount. The editor interface
(`GenerativeImageEditor.edit(image, prompt, seed, params)`) is what a future
Qwen-Image-Edit or LongCat editor implements. It is one class plus one
registry row.

### Fitting FireRed on a 16 GB card

FireRed in bf16 is ~57 GB: a 20B Qwen-Image transformer (41 GB) plus a
Qwen2.5-VL-7B text encoder (16.6 GB). It fits like this:

- **Transformer:** FireRedTeam's own **q4_k_m GGUF** (13.1 GB) from
  `FireRed-Image-Edit-1.1-ComfyUI`. Its tensor names are already diffusers'
  (checked against the file header), which diffusers' `from_single_file`
  needs, because this class has no key conversion.
- **Text encoder:** bitsandbytes **NF4** (~5.5 GB), quantized at load.
- **Model CPU offload:** the components take turns on the GPU. The peak is
  about the transformer plus activations, **~14 GB**.
- **VAE tiling**, plus a resize back to the source's exact pixel size, because
  the labels are for that grid.
- **Lightning:** FireRed's official 1.1 8-step LoRA with Qwen-Image-Lightning's
  scheduler (8 steps, CFG 1), roughly 5× fewer transformer passes.
- **Not offered:** FireRed's "optimized" path (quanto int8 + DBCache +
  compile), which needs ~30 GB by FireRed's own account. On a bigger GPU, pick
  the **bf16** weights (~45 GB).

It **refuses to load** with HTTP 503 and the free-memory figure when less
than ~14 GiB is free (`GENAUG_MIN_FREE_GB`), rather than OOMing mid-load. It
**unloads itself** after `GENAUG_IDLE_UNLOAD_S` (default 300 s), so a
finished studio session hands the GPU back to training.

### Weights

```
python manage.py fetch_genaug_weights            # what's missing + exact commands
python manage.py fetch_genaug_weights --lightning
```

That's ~30 GB: the main repo *minus* its bf16 transformer shards, plus the
GGUF (and the LoRA). As with every other HF model here, download on a machine
with access and `rsync -a` into `data/hf_cache/hub/`.

### The mock editor

`editor = mock` applies classical filters keyed on prompt words (dark, blur,
compress, noise, tint), with no GPU and no weights. It exists to exercise the
studio, validation and builds end to end. Its outputs are labelled `mock` in
file names and manifests.

## Tests

```
# Django side (backend faked)
docker compose run --rm -T -v "$PWD/chachkalica:/app" web python manage.py test genaug --keepdb --noinput
# backend service (mock editor)
docker compose run --rm -T --no-deps genaug-backend python -m unittest ml_backends.genaug.tests_service
# across containers, against a running genaug-backend
docker compose run --rm -T -v "$PWD/chachkalica:/app" -e GENAUG_E2E=1 web python manage.py test genaug.tests_e2e --keepdb --noinput
```

(The service tests need `pip install httpx` in the container first, for
FastAPI's TestClient.)

## Status

Built and tested with the mock editor, in-process and across containers.
**FireRed itself has not generated an image here yet.** Its weights are not
in the cache, and the GPU was busy with training throughout. The loading path
is written against diffusers 0.40 / transformers 5.17 (both import cleanly in
the image), the GGUF's tensor names were verified from its header, and the
recommended settings (40 steps, CFG 4, negative `" "`) come from FireRed's own
`inference.py`. The first real run is the test that is still owed.
