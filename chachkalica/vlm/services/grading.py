"""Score a VLM's free text against a dataset's YOLO labels.

The two sides of a dataset run speak different languages. The ground truth is a
YOLO ``labels/*.txt`` — class ids and boxes — while the model answers in prose.
Nothing here tries to bridge that as *detection*: a VLM produces no boxes, so
the only claim its answer can be checked against is **which classes the image
contains**. So a label file is reduced to the set of class names it mentions, the
answer is scanned for those same names, and the two sets are compared.

Three things about that reduction are load-bearing:

* **Aliases, because a class name is rarely what a model says.** ``classes.txt``
  says ``helmet``; the answer says "hard hat", or just "yes". Every class carries
  a list of extra terms, editable after the fact — see :func:`default_alias_map`.
* **Negation, because "no helmet" contains "helmet".** A prompt phrased as a
  question is answered "No, the worker is not wearing a helmet", which a plain
  substring test scores as *helmet present* — the exact inverse of the truth.
  Mentions preceded by a negation cue in the same clause are therefore read as
  absence claims, which is what makes the metrics mean anything at all for
  question-shaped prompts. It is a heuristic and can misfire, so it is a toggle
  (:func:`predict_classes`'s ``negation``) rather than a law.
* **A missing label file is not an empty one.** No file means *unlabeled* and the
  image is left out of the scoring entirely; an empty file means *nothing is
  present* and is a legitimate negative. ``fleet.services.analytics`` draws the
  same distinction.

Everything here is a pure function over text and sets, which is what lets a
finished run be re-graded with better aliases without paying for the GPU again.
"""

import re

from fleet.reconcile.txt_format import parse_label_text

#: Words that flip a mention from "present" into "absent" when they sit close in
#: front of it. ``n't`` contractions are matched by suffix rather than listed.
_NEGATION_CUES = frozenset({
    "no", "not", "none", "never", "without", "lacks", "lacking", "lack",
    "missing", "absent", "nobody", "cannot", "neither", "nor", "unable", "any",
})

#: How many words in front of a mention are searched for a cue. Five reaches
#: across "no one in this image is wearing a helmet" without reaching over a
#: clause boundary in practice.
_NEGATION_WINDOW_WORDS = 5

#: Clause boundaries a negation is not allowed to reach across, so
#: "no ladder, but everyone has a helmet" does not negate the helmet.
_CLAUSE_BOUNDARY = re.compile(
    r"[.;:!?,]|\b(?:but|however|although|though|whereas|while|yet)\b"
)

_WORDS = re.compile(r"[a-z0-9']+")

#: Separators treated as interchangeable, so a ``hard_hat`` class matches
#: "hard hat", "hard-hat" and "hardhat" alike.
_SEPARATORS = re.compile(r"[\s_\-]+")


def default_alias_map(classes) -> dict[str, list[str]]:
    """The starting alias table: every class, no extra terms.

    An empty list is not "match nothing" — the class's own name is always a
    term (:func:`_terms_for`). This exists so the re-grade editor has a row per
    class to type into.
    """
    return {name: [] for name in classes}


def _terms_for(name: str, alias_map: dict | None) -> list[str]:
    terms = [name]
    for alias in (alias_map or {}).get(name) or []:
        alias = (alias or "").strip()
        if alias:
            terms.append(alias)
    return terms


def _pattern_for(term: str):
    """A word-boundaried pattern for one term, or None if it is empty.

    Separators inside the term become optional so one alias covers the whole
    "hard hat"/"hard-hat"/"hardhat" family, and a trailing plural is accepted.
    The boundaries are lookarounds rather than ``\\b`` so that a term still
    matches when it is glued to punctuation or an underscore.
    """
    parts = [re.escape(p) for p in _SEPARATORS.split(term.strip().lower()) if p]
    if not parts:
        return None
    body = r"[\s_\-]*".join(parts)
    return re.compile(rf"(?<![a-z0-9]){body}(?:e?s)?(?![a-z0-9])")


def compile_matchers(classes, alias_map: dict | None = None) -> list[tuple[str, list]]:
    """``[(class name, [compiled patterns])]`` for one grading configuration."""
    matchers = []
    for name in classes:
        patterns = [p for p in (_pattern_for(t) for t in _terms_for(name, alias_map)) if p]
        matchers.append((name, patterns))
    return matchers


def _is_negated(text: str, start: int) -> bool:
    """Whether the mention at ``start`` is preceded by a negation cue.

    Only the words of the mention's own clause count, and only the last few of
    them — a cue two sentences back says nothing about this mention.
    """
    clause_start = 0
    for boundary in _CLAUSE_BOUNDARY.finditer(text, 0, start):
        clause_start = boundary.end()
    words = _WORDS.findall(text[clause_start:start])[-_NEGATION_WINDOW_WORDS:]
    return any(word in _NEGATION_CUES or word.endswith("n't") for word in words)


def predict_classes(text: str, matchers, *, negation: bool = True) -> list[str]:
    """The classes an answer claims are present, in ``matchers`` order.

    A class is claimed when at least one of its terms appears un-negated. A
    mention that is *only* ever negated does not count as present — that is the
    whole point of the negation pass — but one un-negated mention is enough,
    because "a helmet, but no vest" asserts the helmet.
    """
    lowered = (text or "").lower()
    if not lowered:
        return []

    predicted = []
    for name, patterns in matchers:
        for pattern in patterns:
            hit = False
            for match in pattern.finditer(lowered):
                if not negation or not _is_negated(lowered, match.start()):
                    hit = True
                    break
            if hit:
                predicted.append(name)
                break
    return predicted


def ground_truth_classes(label_text: str, classes) -> list[str]:
    """The class names a YOLO label file annotates, deduplicated, in file order.

    Class ids outside ``classes.txt`` are dropped rather than raising — the same
    tolerance ``analytics.analyze_dataset`` shows, since a stale label file must
    not be able to abort a whole run.
    """
    _w, _h, objects = parse_label_text(label_text or "")
    seen, names = set(), []
    for obj in objects:
        index = obj["class_id"]
        if 0 <= index < len(classes) and index not in seen:
            seen.add(index)
            names.append(classes[index])
    return names


def is_exact_match(gt_classes, predicted_classes) -> bool:
    """Whether the answer named exactly the classes the label file annotates.

    Set equality, so an image labeled ``helmet`` whose answer names *helmet and
    vest* is not a success: the extra claim is wrong even though the label was
    found. Per-class precision/recall in :func:`metrics` is what shows which
    half of a mismatch went wrong.
    """
    return set(gt_classes) == set(predicted_classes)


def _prf(tp: int, fp: int, fn: int) -> dict:
    """Precision/recall/F1 with the usual zero-division convention.

    All three are None only when the class took no part in the evaluation at
    all — never labeled, never claimed — which is what keeps such a class out of
    the macro average instead of scoring it a misleading zero. A class that was
    labeled but never claimed genuinely scores zero, and must count.
    """
    if not (tp or fp or fn):
        return {"precision": None, "recall": None, "f1": None}
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    denominator = precision + recall
    f1 = 2 * precision * recall / denominator if denominator else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _round(value):
    return None if value is None else round(value, 4)


def _confusion(rows, classes) -> dict:
    """A GT-class × answer table over the images with exactly one labeled class.

    Restricted to single-class ground truth because that is the only case where
    "the model said the wrong thing" has one meaning. Answers that name several
    classes, or none, get their own columns instead of being silently dropped —
    on a two-class dataset those two columns are usually where the story is.
    """
    columns = list(classes) + ["(none)", "(multiple)"]
    index = {name: i for i, name in enumerate(columns)}
    grid = {name: [0] * len(columns) for name in classes}
    total = 0

    for gt, predicted in rows:
        if len(set(gt)) != 1:
            continue
        actual = next(iter(set(gt)))
        if actual not in grid:
            continue
        predicted = set(predicted)
        if not predicted:
            column = "(none)"
        elif len(predicted) == 1:
            column = next(iter(predicted))
        else:
            column = "(multiple)"
        if column not in index:
            continue
        grid[actual][index[column]] += 1
        total += 1

    if not total:
        return {}
    return {
        "columns": columns,
        "rows": [
            {"actual": name, "counts": grid[name], "total": sum(grid[name])}
            for name in classes
            if sum(grid[name])
        ],
        "total": total,
    }


def metrics(results, classes) -> dict:
    """Aggregate scored rows into the numbers the analytics panel renders.

    ``results`` is any iterable of objects carrying ``has_label``,
    ``gt_classes`` and ``predicted_classes`` — the stored result rows, or plain
    stand-ins in tests. Only labeled rows are scored; the rest are counted and
    reported so a half-labeled dataset cannot quietly inflate an accuracy.
    """
    per_class = {name: {"tp": 0, "fp": 0, "fn": 0, "support": 0} for name in classes}
    graded = exact = unlabeled = 0
    confusion_rows = []

    for result in results:
        if not result.has_label:
            unlabeled += 1
            continue
        gt = set(result.gt_classes or [])
        predicted = set(result.predicted_classes or [])
        graded += 1
        if gt == predicted:
            exact += 1
        confusion_rows.append((gt, predicted))
        for name in classes:
            bucket = per_class[name]
            in_gt, in_pred = name in gt, name in predicted
            if in_gt:
                bucket["support"] += 1
            if in_gt and in_pred:
                bucket["tp"] += 1
            elif in_pred:
                bucket["fp"] += 1
            elif in_gt:
                bucket["fn"] += 1

    class_rows = []
    for name in classes:
        bucket = per_class[name]
        scores = _prf(bucket["tp"], bucket["fp"], bucket["fn"])
        class_rows.append({
            "name": name,
            **bucket,
            "precision": _round(scores["precision"]),
            "recall": _round(scores["recall"]),
            "f1": _round(scores["f1"]),
        })

    tp = sum(row["tp"] for row in class_rows)
    fp = sum(row["fp"] for row in class_rows)
    fn = sum(row["fn"] for row in class_rows)
    micro = _prf(tp, fp, fn)

    # Macro averages only over classes that actually occur, so a class the
    # dataset never labels cannot drag the average to zero.
    present = [row for row in class_rows if row["precision"] is not None]
    def _mean(key):
        values = [row[key] for row in present if row[key] is not None]
        return round(sum(values) / len(values), 4) if values else None

    return {
        "graded_images": graded,
        "unlabeled_images": unlabeled,
        "exact_matches": exact,
        "exact_match_accuracy": _round(exact / graded) if graded else None,
        "micro": {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": _round(micro["precision"]),
            "recall": _round(micro["recall"]),
            "f1": _round(micro["f1"]),
        },
        "macro": {
            "precision": _mean("precision"),
            "recall": _mean("recall"),
            "f1": _mean("f1"),
        },
        "per_class": class_rows,
        "confusion": _confusion(confusion_rows, classes),
    }
