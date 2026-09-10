# Marketing Studio

## Context

[Marketing renders](marketing-videos.md) added a second action to the Videos
tab: same model, same pipelines, same job row, plus a Look section and a frame
preview. That was the right first move — the machinery was already there — but
it left the feature living inside `videos/`, sharing a tab, a video library and
a model picker with a section aimed at a different job. The detection Videos
tab is where datasets get made and models get checked; a marketing render is
someone cutting a clip for a deck.

**Marketing Studio** is that second job given its own surface. Three tabs, one
action, and its own everything: its own video library, its own bundle set, its
own render log, its own presets.

It does **not** replace the Videos tab's marketing action, which stays exactly
as it was. Both exist; they simply do not share state.

## What it deliberately does not have

- **No trained-model or exported-artifact source.** A render here always runs an
  **infer bundle** (see [infer bundles](infer-bundles.md)). That is the whole
  reason the form has no first step: there is nothing to ask before showing the
  real page.
- **No frame extraction.** Making datasets is the Videos tab's job.
- **No plain overlay.** Every row in this section's log is a look; there is no
  `render_style={}` half to split off, and so no "Marketing videos" sub-tab
  either — the split the Videos tab needs does not arise here.

## Where things live

Two new directories, both on the `FleetSettings` singleton (Fleet → Fleet
settings), beside `videos_dir` and `vlm_videos_dir`:

| Setting | Default | What it holds |
|---|---|---|
| `marketing_videos_dir` | `data/marketing/videos` | The studio's clips, plus `inferred/` beneath it for finished renders |
| `marketing_bundles_dir` | `data/marketing/bundles` | The bundles this section offers |

Both defaults sit under `data/`, and they must stay somewhere under it: that is
the directory compose mounts into the trainer container at the same path
(`./chachkalica/data:/app/data`), and a bundle's model reaches `/predict_image`
as an absolute path. A root outside that mount fails at render time with a
file-not-found from inside the trainer.

**The separate video root is load-bearing, not tidiness.** The render loop is
shared code (`videos/services/inference.py`), and two things in it are keyed by
directory: `unique_output_filename` only collision-checks the folder it is
given, and the per-job scratch frame is named `.<pk>_frame.jpg`. Two tables
have overlapping pks, so one shared `inferred/` folder would let two concurrent
renders corrupt each other's frames mid-encode. Hence the `root=` parameter
those functions grew, and hence `marketing_videos_dir` must never be pointed at
`videos_dir`.

**The separate bundle root is a curation choice.** This section does not read
`TrainingSettings.bundles_root` at all. It has its own "Sync bundle" endpoint
(`admin:marketing_studio_video_bundle_sync`) rather than the project-wide
`bundle-sync` one, because that view passes no settings object and would
therefore always resolve a relpath against Training's root — syncing the wrong
geometry onto the render. The retargeting itself is
`marketing_studio/services/paths.py::_BundleSettings`, a `TrainingSettings`
stand-in that overrides `bundles_root` and delegates everything else, so
`validate(load_test=True)` still reaches the real trainer.

## What it shares with `videos/`

Code, never state. Imported unchanged:

- `videos/services/render_style.py` — the whole look system: the knob tables,
  `parse_form`/`normalize`/`form_values`, and `MarketingRenderer`. One
  renderer, so a fix to how a box is drawn lands in both sections at once.
  [The knob-by-knob reference is in the marketing renders doc](marketing-videos.md).
- `videos/services/inference.py` — the ffmpeg render loop and the preview
  cache, reached through the `root=` parameter described above.
- `videos/services/downloader.py`, `videos/services/streaming.py`.
- `videos/static/videos/marketing_style.js` and
  `training/static/training/bundle_sync.js` — both are driven entirely by DOM
  data attributes, so they serve this section's pages with no changes. They are
  deliberately *not* copied: `staticfiles` is first-app-wins, and a second copy
  under a colliding name would silently serve one app's version to both.

Not shared, by design:

- **Tables.** `marketing_studio.Video`, `.Render`, `.RenderPreset` are their own.
- **Presets.** A look saved here does not appear on the Videos tab's Render
  presets, and the same preset name may exist on both sides holding different
  styles. If someone reports that as a bug, it isn't.
- **Videos.** A clip imported here is a separate row from the same clip on the
  Videos tab, even if both point at a file with the same name.

## Using it

1. Point `marketing_bundles_dir` at a directory holding at least one bundle
   (copy one in, or point it at an export's `…-bundle/`).
2. **Marketing Studio → Videos → Add**: import a file already sitting in
   `marketing_videos_dir`, or paste a link to download. "Import every new file
   in the studio's videos folder" picks up anything dropped in by hand.
3. Select one clip → **Run model inference for marketing…**. The form opens
   directly on the bundle select.
4. Pick a bundle, press **Sync bundle**. It reads the manifest, checks the
   model/detector/classes, fills the pipeline fields and locks them. The
   optional load test proves the artifacts load on this machine — worth it for
   a TensorRT engine that arrived from somewhere else, and it briefly takes the
   trainer's GPU.
5. Dial the look, press **Preview** to render one frame through the exact
   renderer the video will use. Style changes redraw from detections already
   fetched, so only moving the position slider costs GPU time.
6. **Save as preset** to keep a look by name (upsert — saving under an existing
   name amends it).
7. **Run** queues the render; it appears under **Inferred videos** with a play
   and download link when it finishes.

## Gotchas

- Both sections share the single `default` RQ queue, so a studio render and a
  Videos-tab inference serialize behind the same workers.
- The pipeline geometry is re-read from the bundle when the render is
  submitted, not just when Sync is pressed — a stale page or a bundle that
  changed on disk still runs what the bundle says today.
- The section's roots are not created for you. `inferred/` is made on demand;
  the library directory is created by the first download, but an operator
  importing by hand needs the directory to exist.
