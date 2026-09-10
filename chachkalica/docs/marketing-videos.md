# Marketing renders ("Run model inference for marketing…")

## Context

`videos/` already had one way to put a model's output on a clip: **Run model
inference…**, which runs the model frame by frame and burns boxes on with
`videos/services/inference.py::draw_boxes` — one line width, one label format,
one colour per class name by md5. That overlay exists to answer *what did the
model do*, and the live camera path (`cameras/services/live_inference.py`)
draws with the same function, so it is deliberately fixed.

The other question people keep asking of the same machinery is *can we show
this to someone* — a clip for a deck, a landing page, a customer demo. That one
is not served by a debugging overlay: it needs boxes that look designed, labels
that don't flicker, a resolution that can be emailed.

**Run model inference for marketing…** is that second action. Same model
sources, same pipelines, same job row, same queue — plus a Look section and a
frame preview.

There is now also a [Marketing Studio](marketing-studio.md) section: the same
renderer and the same Look section, given its own video library, its own bundle
root and its own presets, and narrowed to infer bundles only. This action stays
exactly as documented below; the two share code but no state.

## What it adds

Everything below is one JSON blob on the job (`InferenceJob.render_style`), read
back by `videos/services/render_style.py::MarketingRenderer`. An empty
`render_style` (which is what the plain action always writes) means the old
overlay, unchanged.

| Group | Knobs |
| --- | --- |
| Boxes | outline style (12 of them, below), thickness, corner radius, bracket length, dash length, dot spacing, double-line gap, chamfer size, glow spread, sketch jitter, outline opacity, translucent fill, dark halo |
| Colour | per-class palette (6 curated), one fixed colour, or a low→high confidence gradient; per-class hex overrides |
| Labels | on/off, text (class / confidence / both), chip style (pill / bar / plain / outlined), position, text colour, font, size, chip opacity, decimals, uppercase |
| What to show | class allow-list, **label every box as one class**, minimum box area, cap on boxes per frame |
| Scene | spotlight (dim everything outside the boxes), per-class counter in a corner, watermark text + corner + colour + opacity + size |
| Motion | smoothing, fade in/out, outline animation + period |
| Output file | delivery height, x264 CRF |

### The outlines

| | |
| --- | --- |
| `solid` / `rounded` | the plain rectangle, sharp or soft-cornered |
| `corners` | four corner marks — points at the subject without boxing it in |
| `crosshair` | brackets plus a tick at the middle of each edge and a centre cross; the "acquiring target" look |
| `dashed` / `dotted` | broken line or round beads, both walked around the perimeter so the corners stay in step |
| `double` | two concentric lines — reads as deliberate framing rather than a machine's output |
| `chamfer` | an octagon, corners cut off; HUD panel |
| `glow` | a neon bloom: three translucent passes fanning outwards plus a bright core (cv2 has no additive blending, so stacked blended passes are how you fake one) |
| `sketch` | a doubled, wobbling rectangle. The wobble is seeded from the **class name**, not the frame or the box position — a jitter reseeded per frame boils like video noise instead of looking drawn |
| `underline` | a bar under the subject with two short lifts, and nothing else; the least surveillance-looking of the set |
| `none` | fill and label only |

`animation` layers over any of them: **marching** crawls the dashes or dots
around the box (phase comes from the rendered-frame counter, so a re-render
animates identically), **pulse** breathes the line thickness. Both are invisible
in the preview, which renders `still=True` — a single frame has no motion.

Four of those are worth explaining.

**Spotlight** darkens the whole frame and pastes the box interiors back from a
copy. Cheaper than masking at 4K, and it is what makes a busy scene read as
"look here".

**Motion** is the only part with state. Raw per-frame model output jitters by a
few pixels and drops out for the odd frame — invisible on a debugging overlay,
glaring in something people watch. `_Tracker` matches this frame's boxes to last
frame's by class, then overlap, then proximity (a subject can move most of its
own width between updates, especially with `frame_stride > 1`, and a track that
fails to match is a box that fades out under its own replacement). Matched
tracks ease toward their new position; new ones fade up; vanished ones fade out
over a bounded number of frames.

Neither knob invents a detection: smoothing moves boxes the model returned, and
a fade-out is capped by `fade_frames` (set it to 0 to draw exactly what each
frame's output says). *Score threshold*, on the pipeline half of the form, is
still the only thing that decides what counts as a detection.

**Glow's drawing ROI** is sized from the widest stroke pass rather than from the
nominal line thickness. Each box is drawn into its own small region of the frame
(so the alpha compositing copies a box-sized patch, not a 4K one), and a bloom
fans much further out than its line does — sizing that region by thickness clips
the glow into a square.

**Label every box as** (`force_class`) is the one knob in the Look section that
changes what a box *says* rather than how it looks. Pick a class and every box
drawn is labelled and coloured as that class, whatever the model returned —
positions and confidences untouched. It is for the clip whose class names are
not the ones to show an audience: a model that says `person_no_helmet` in a deck
about `violation`, or a two-class detector whose second class is noise in this
particular shot.

Three things keep it honest and predictable:

* It applies **last**, after the allow-list, the size floor and the cap — so
  *Only these classes* still selects on what the model actually returned.
* It is **drawing only**. The job's stored detections, the per-class counts and
  the score threshold are the model's own, unrelabelled; the Inferred videos row
  says `all "<class>"` beside the look so a relabelled render can't be mistaken
  for the model's own labels.
* `_Tracker` still matches on the model's class (`render_style.true_class`,
  which reads the real class parked aside by the relabel). Matching on the
  relabel would make every box on the frame a candidate for every other, and a
  helmet's track would ease onto the vest that replaced it.

Its dropdown is the only field on the form whose vocabulary is the *model's*
rather than this module's: the classes come from the bundle manifest, the
catalogued model's row, or the sidecar beside the exported artifact
(`exports.read_class_names`), and the JS refills it when a different model is
picked. A value the selected model has no class for is kept rather than dropped
— it is only ever drawn, so a relabel carried over from the last render or
loaded from a preset survives.

**Delivery height** resizes each frame *before* drawing, not with an ffmpeg
filter afterwards. Style sizes are in output pixels, so drawing a 4K frame and
then squeezing it to 1080p would hand back a third of the stroke that was asked
for. Inference still sees the full-resolution frame.

## The preview

Thirty knobs are unusable if seeing their effect means re-encoding a five-minute
clip, so the form renders **one frame** through the very same
`MarketingRenderer` and shows it beside the fields:

```
form (POSTed whole, unmodified)
        │
        ▼
admin/videos/marketing-preview/       VideoAdmin.marketing_preview_view
        │  _build_inference_job(...)  ← the same validation the submit does
        ▼
inference.preview_frame(job, position)
        │
        ├── seek, decode one frame
        ├── boxes: <videos_root>/inferred/.preview_cache/<sha>.json
        │      hit  → no model call
        │      miss → POST /predict_image, then cache
        ├── resize to min(delivery height, source, 1080p)
        └── MarketingRenderer(style, still=True).draw(...)   → jpeg → data URI
```

Two consequences worth knowing:

* **Style changes are free.** The cache key is the exact `/predict_image`
  payload + the video's size/mtime + the frame index, so re-drawing the same
  frame with a different look never touches the GPU. Moving the position slider,
  changing the model or the pipeline, or pressing *Re-run the model* does.
* **`still=True` drops smoothing and fading.** A single frame has no history, so
  a preview would otherwise show every box part-way through its fade-in. Those
  two knobs are the ones you set by eye on the finished clip.

The form arrives carrying the last marketing style used on this instance, so a
house look, once dialled in, is one press away on the next clip.

## Presets

The last-used style covers "another one like that one". **Presets** cover "we
have three looks and I pick between them": a `RenderPreset` row is a named style
dict, saved by the *Save this look* button at the bottom of the form and offered
from the *Preset* dropdown at the top.

* Saving is an **upsert by name** — saving over an existing preset is how you
  amend one after tweaking it, and loading a preset prefills the name box so
  that is the default outcome.
* Presets hold the **look only** — not the model, pipeline or detector. Those
  come from the model's own recorded metadata (`pipeline-metadata.md`), and a
  preset that carried them would quietly undo that convention.
* Loading is client-side and instant: the server sends each preset already in
  the form's shape (`{"style_thickness": 3, …}` — `render_style.form_values`),
  so the page applies one by assigning each value to the input of that name and
  never learns what a knob is. The preview then re-renders from the cached
  detections, so trying a preset costs nothing.
* Presets are normalized on the way out, so one saved before a knob existed
  fills in the missing defaults rather than breaking the form.
* Rename and delete them under **Render presets** in the Videos app — an
  ordinary ModelAdmin, which is also where you can read exactly what a look is.

## Where things live

| | |
| --- | --- |
| `videos/services/render_style.py` | the style vocabulary (field tables, palettes), `normalize` / `parse_form`, `MarketingRenderer`, `_Tracker` |
| `videos/services/inference.py` | `preview_frame`, and the render loop picking renderer + encode options |
| `videos/admin.py` | `run_inference_marketing`, `_inference_wizard` (shared with the plain action), `marketing_preview_view`, `render_preset_save_view`, `RenderPresetAdmin` |
| `videos/models.py` | `InferenceJob.render_style`, `RenderPreset` |
| `templates/admin/videos/run_inference_marketing.html` | the Look section + preview pane |
| `templates/admin/videos/_inference_fields.html` | the model/pipeline half, shared verbatim by both actions |
| `videos/static/videos/marketing_style.js` | range read-outs, conditional rows, the preview fetch, preset load/save |

Adding a knob is a row in one of the `_*_FIELDS` tables in `render_style.py`
(which gives it parsing, validation, defaults and storage), a field named
`style_<knob>` in the template, and the drawing code that reads it. Nothing in
the admin or the JavaScript knows the name of a single style knob — the form is
POSTed whole to the submit, the preview and the preset-save endpoint alike, and
all three parse it with the same function. The one exception is `force_class`,
whose *options* have to be fetched per model: `_class_names_by_model` in
`videos/admin.py` builds them and `marketing_style.js` refills the select from
them. The value itself still goes through the same parse as everything else.

Adding an **outline** is an entry in `BOX_STYLE_CHOICES`, a drawing function
beside the others, and a branch in `MarketingRenderer._draw_one`'s `stroke()`.
If it fans out past its line width, say so in `_stroke_passes` too, or the
drawing ROI will clip it.

## Gotchas

* A knob added to `_inference_fields.html` reaches **both** actions — that is
  the point (they must stay the same run), but it does mean the plain form
  changes too.
* Hidden rows still submit. The renderer ignores knobs its style doesn't use, and
  keeping the value means flipping back to "dashed" restores what you had.
* A relabel (*Label every box as*) is presentation only, and deliberately does
  not touch the class allow-list, the counts, or the stored detections. If a
  clip needs its boxes to *be* another class rather than to look like one, that
  is a different model, not this knob.
* Class colours are hashed from the class *name*, not its index, and are
  independent of what else is in the frame — so a preview and the video it
  previews agree, and two runs of a clip match. Pin a colour exactly with a
  per-class override.
* The preview cache lives under `<videos_root>/inferred/.preview_cache/` and is
  pruned to the newest 200 entries. Deleting it costs nothing but a re-run.
