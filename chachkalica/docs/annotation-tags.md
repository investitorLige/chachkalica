# Annotation tags

Extra questions annotators answer beyond drawing boxes: *is this frame indoors,
is this box occluded, how far away is it*. Defined per dataset, compiled into
the labeling interface of every Label Studio project created for it.

## Where the editor is

A project's labeling interface is fixed the moment the project is created, so
the editor sits in the two actions that create projects — it is their
confirmation step:

- Datasets → **Set up selected dataset(s) for all active annotators**
- Datasets → **Set up + sync selected dataset(s) for one annotator…**

Settle the tags, hit *Set up projects*, and the projects are created asking for
them. Leave both sections empty for boxes only, exactly as before.

Tags belong to one dataset, so selecting several turns the editor into a
read-only summary of what each already carries; they are set up unchanged.

Datasets → **Edit annotation tags (frame-wide / box-wide)…** is the same editor
on its own, for going back to the tags later without setting anything up.

## Two scopes

The difference is only how often the question is asked.

| Scope | Asked | Where the annotator sees it |
|---|---|---|
| **Frame-wide** | once per image | under the image on the canvas, under its own heading |
| **Box-wide** | once per region drawn | in the details panel of the selected region |

A box-wide tag is a plain control with `perRegion="true"`, which is what moves
it off the canvas and into the region panel.

## Widgets

All four list widgets are the same `<Choices>` tag differing by attribute —
which is why the widget is a dropdown in the editor rather than a separate kind
of tag.

| Widget | Emits | Annotator gets |
|---|---|---|
| checkboxes | `<Choices choice="multiple">` | tick any number |
| radio buttons | `<Choices choice="single">` | pick exactly one |
| dropdown | `<Choices choice="single" layout="select">` | pick one, from a select |
| multi-select dropdown | `<Choices choice="multiple" layout="select">` | pick any, from a select |
| star rating | `<Rating maxRating="N">` | 1–N stars |
| free text | `<TextArea>` | types anything |

Marking a tag **required** makes Label Studio refuse the submit until it is
answered — for a box-wide tag, on every region.

## The name is not just a key

Label Studio labels a `perRegion` control in the region panel with its own
`name`, so a box-wide tag's name is what the annotator reads. Frame-wide tags
get a real `<Header>` above them instead, because they render on the canvas
where there is room for one.

Box-wide tags deliberately get **no** header. `<Header>` is not perRegion-aware:
one emitted next to a per-region control would stay behind on the canvas as a
caption for a control that isn't there.

A name must start with a letter and contain only letters, digits, `_` and `-`.
It also cannot be `image`, `bbox`, `segmentation` or `sam_point` — the generated
config already uses those, and two controls sharing a `name` is what Label
Studio reads as *one* control, so a tag called `bbox` would silently merge into
the box tool.

## Editing tags after the projects exist

Tags are baked into a project's interface when the project is created, and
creation is idempotent on the project title — running setup again skips a
project that exists rather than rebuilding it. So an edit does not reach one on
its own. Tick **also push onto the N projects that already exist** (present on
all three pages) and the interface is rewritten in place, keeping every
annotation already made — the alternative, recreating the project, would throw
that work away.

The push is best-effort per annotator: a container that is down is skipped, and
Label Studio rejects a change that would orphan existing regions (removing a tag
annotators have already used can fail). Either way the outcome lands on that
annotator's row in **Projects**, not on the others.

## Where the answers end up

Sync writes them into the label directory, as one `annotation_tags.json` sitting
beside the `.txt` files it describes:

```
target/<dataset>/<annotator>/annotation_tags.json   # written by fleet sync
source/<dataset>/labels/annotation_tags.json        # after "Promote … to source labels"
```

A sidecar rather than extra columns on the label lines, because the `.txt` has
no room: any extra number on a line is read back as a polygon coordinate, so
appending a tag value there would silently corrupt the boxes for every consumer
— this app's own parser, the COCO build, and the trainer's dataloader alike.
Nothing globs a label directory for anything but `*.txt`, so the sidecar is
inert to all three.

```json
{
  "version": 1, "dataset": "ppe", "annotator": "alice",
  "tags": [{"name": "weather", "scope": "frame", "widget": "radio",
            "choices": ["sun", "rain"], "max_rating": 5}],
  "images": {
    "img01.jpg": {
      "frame": {"weather": ["rain"]},
      "boxes": [{"row": 0, "class_id": 2, "tags": {"occluded": ["heavy"]}}]
    }
  }
}
```

Each widget keeps its natural type — a list for the four list widgets, a number
for a rating, a string for free text. An unanswered tag is absent, which is not
the same as an empty answer.

**`row` is the box's line in that image's `.txt`**, counting from 0 after the
`W H` header. It is the only identity a geometry-only format leaves to join on,
and it is assigned in the same pass that writes the file — which matters,
because a region Label Studio has but the `.txt` does not (an unknown class, a
degenerate box) shifts the line of every box after it. `class_id` is stored
beside the row so a reader can *detect* a later drift rather than silently
attribute an answer to the wrong box.

**Sync owns this file, not the webhook.** The webhook writes one image's `.txt`
per annotation event; this is one file describing a whole project, and a
per-event read-modify-write of it would race itself across webhook threads and
processes. So tag answers land at sync, exactly like the COCO document beside
them. Sync also deletes a stale sidecar when a dataset's tags are all removed.

"Promote an annotator's annotations to source labels" copies the sidecar across
with the labels (copies, not moves — the annotator's own output stays
self-describing). Promoting an annotator who has no tag answers *removes* the
sidecar from source rather than leaving the previous annotator's rows pointing
at somebody else's boxes.

The answers are still in Label Studio too: export the project as JSON from there
(or **Fleet → Projects**), where each tag is a result whose `from_name` is the
tag's name.

## What they are for

[Tag analytics](tag-analytics.md) re-scores any eval over the slices these
answers define — per tag value, as a cross-tab of two tags, or over an arbitrary
intersection — without re-running the model.

## Preview

The standalone editor prints the full labeling interface a new project for the
dataset would get, classes and drawing tools included. Label Studio also renders its own
live preview under *Settings → Labeling Interface* in the project — worth a look
before handing a project to annotators, since layout is the one thing the
generated XML can't promise.
