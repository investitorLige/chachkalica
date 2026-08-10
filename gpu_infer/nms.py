"""Class-aware overlap NMS on the device, exactly reproducing the greedy CPU original.

``chachak.boxes._class_aware_overlap_nms`` is greedy with a **shrinking candidate set**: take
the highest-scoring box still in play, keep it, delete everything it duplicates, repeat. Two
consequences matter here.

First, ``torchvision.ops.batched_nms`` cannot stand in. The duplicate test is
``iou >= t`` **or** ``containment >= t``, where containment divides the intersection by the
*smaller* box's area. That second term is what catches the near-containment duplicates
overlapping tiles and overlapping person crops produce — a small box sitting almost entirely
inside a large one, where IoU stays low. Standard NMS keeps both.

Second, greedy suppression is inherently sequential: a suppressed box can never suppress
anything, so the answer depends on the order decisions are made. It is the
lexicographically-first maximal independent set on the suppression DAG, which has no
fixed-depth parallel form in general. What it *does* have is a round-based form that is
**exactly** equivalent, described in :func:`class_aware_overlap_nms`.

Why this runs on compacted detections
-------------------------------------

Unlike the rest of this package, NMS is not run over padded slots. It needs a pairwise
``[M, M]`` relation, and ``M`` for a padded buffer is ``max_crops x K`` — at the eval score
floor of 0.001 an rfdetr stage-2 keeps most of its 300 queries per crop, so ``M`` reaches a few
thousand and each ``[M, M]`` **float** intermediate is tens of megabytes. Compacting first makes
``M`` the real detection count, which is tens.

That costs nothing, because the compaction is the synchronization this stage was going to
perform anyway: NMS is the last thing before results leave for the host. Being precise about the
budget — one synchronization per *stage boundary*, not one per batch: the two-stage pipelines
(``people_detect_first``, ``batch_people``) pay one after stage 1, because person-crop bounds
have to be host integers for the slice to exist at all, and one at the end — so two per frame
against the original's ~14. Single-stage ``batch_detect`` pays one.

The suppression matrix is still built in row chunks and reduced to ``bool`` immediately, so peak
float memory stays bounded even when a pathological frame does produce thousands of detections.
"""

from __future__ import annotations

from typing import Optional, Tuple

#: Rows processed per chunk when building the pairwise relation. Bounds the float intermediates
#: at ``chunk x M`` rather than ``M x M``; the bool result is kept in full.
_ROW_CHUNK = 1024


def _areas(boxes):
    """Box areas, clamped exactly as the original does before any ratio is taken."""
    return (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (
        boxes[:, 3] - boxes[:, 1]
    ).clamp(min=0)


def _outranks(scores):
    """``[M, M]`` bool: does row ``i`` come before row ``j`` in the original's processing order?

    The original walks ``scores.argsort(descending=True)``, so ``i`` outranks ``j`` when it
    scores higher, and ties fall to whatever the sort does with them. Deriving the relation
    directly as ``(s_i > s_j) | ((s_i == s_j) & (i < j))`` is precisely a **stable** descending
    argsort, needs no sort at all, and is deterministic across devices.

    **The one documented parity gap.** ``argsort(descending=True)`` is not stable unless asked,
    so the original's behaviour on two same-class overlapping boxes with *bit-identical* scores
    is implementation-defined and can differ between the CPU it runs on and the CUDA this runs
    on. This picks the lower index, which is what a stable sort would do. Real float32 detection
    scores essentially never tie — and the one systematic source of ties, the plugin's
    zero-padded tail, is masked out before this point — but a synthetic test can hit it, so
    ``tests/test_nms.py`` documents it as a known difference rather than pretending it away.
    """
    import torch

    index = torch.arange(scores.shape[0], device=scores.device)
    higher = scores.unsqueeze(1) > scores.unsqueeze(0)
    tied = scores.unsqueeze(1) == scores.unsqueeze(0)
    earlier = index.unsqueeze(1) < index.unsqueeze(0)
    return higher | (tied & earlier)


def suppression_matrix(boxes, scores, classes, threshold: float):
    """``[M, M]`` bool where entry ``(i, j)`` means "``i`` suppresses ``j``".

    ``i`` suppresses ``j`` when it outranks ``j``, carries the same class, and overlaps it by
    either measure. Built in row chunks: the float intermediates (intersection, union, the two
    ratios) exist only for ``_ROW_CHUNK`` rows at a time, while the bool result is retained in
    full — a bool ``[M, M]`` is an eighth the size of one float one.

    Every arithmetic step mirrors ``_class_aware_overlap_nms``'s inner block, including the
    ``clamp(min=eps)`` guards on both denominators and the fact that both divisions are by a
    tensor.
    """
    import torch

    count = boxes.shape[0]
    if count == 0:
        return torch.zeros((0, 0), dtype=torch.bool, device=boxes.device)

    epsilon = torch.finfo(boxes.dtype).eps
    areas = _areas(boxes)
    outranks = _outranks(scores)
    same_class = classes.unsqueeze(1) == classes.unsqueeze(0)

    suppresses = torch.zeros((count, count), dtype=torch.bool, device=boxes.device)
    for start in range(0, count, _ROW_CHUNK):
        stop = min(start + _ROW_CHUNK, count)
        rows = slice(start, stop)

        x1 = torch.maximum(boxes[rows, 0].unsqueeze(1), boxes[:, 0].unsqueeze(0))
        y1 = torch.maximum(boxes[rows, 1].unsqueeze(1), boxes[:, 1].unsqueeze(0))
        x2 = torch.minimum(boxes[rows, 2].unsqueeze(1), boxes[:, 2].unsqueeze(0))
        y2 = torch.minimum(boxes[rows, 3].unsqueeze(1), boxes[:, 3].unsqueeze(0))
        intersection = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

        union = areas[rows].unsqueeze(1) + areas.unsqueeze(0) - intersection
        iou = intersection / union.clamp(min=epsilon)
        smaller = torch.minimum(areas[rows].unsqueeze(1), areas.unsqueeze(0))
        contained = intersection / smaller.clamp(min=epsilon)

        suppresses[rows] = (
            same_class[rows] & ((iou >= threshold) | (contained >= threshold))
        )
    # A box never suppresses itself or anything that outranks it.
    return suppresses & outranks


def class_aware_overlap_nms(
    boxes, scores, classes, threshold: float, *, rounds: int = 8, strict: bool = True
) -> Tuple["object", "object"]:
    """Greedy class-aware overlap NMS. Returns ``(keep_indices, converged)``.

    ``keep_indices`` is int64, ordered by **descending score** — the same order the original's
    ``keep`` list ends up in, so ``dets[keep]`` reproduces its row order and not merely its row
    content. ``converged`` is a device-side bool scalar: ``True`` when every box was decided
    within ``rounds``.

    The round-based equivalence
    ---------------------------

    Maintain ``undecided`` (initially every box) and ``kept`` (initially none). Each round:

    1. A box whose every suppressor is already **decided** is kept. Because step 2 removes a
       kept box's victims in the same round it is kept, any box still undecided has no *kept*
       suppressor — so all its decided suppressors were themselves suppressed, and nothing can
       stop it. The globally top-ranked box has no suppressors at all, so round one always
       decides at least that.
    2. Everything those newly-kept boxes suppress becomes decided-and-not-kept.

    At least one box is decided per round (the top undecided box can have no undecided
    suppressor, since a suppressor must outrank it), so this terminates. The trip count is the
    longest suppression *chain*, not the box count: A suppresses B, and B would have suppressed
    C, so greedy keeps C — that chain is depth two. Real detections at IoU >= 0.5 bottom out
    around two to four, hence the default of eight.

    The loop runs a **fixed** number of rounds rather than testing for convergence, because a
    per-round ``.any()`` test drains the CUDA queue — one such test costs more than the whole
    NMS and forfeits exactly the overlap this package exists to create. ``strict`` instead
    returns the convergence flag for the caller to fold into a synchronization it is already
    performing.
    """
    import torch

    count = boxes.shape[0]
    if count == 0:
        return (
            torch.zeros((0,), dtype=torch.int64, device=boxes.device),
            torch.ones((), dtype=torch.bool, device=boxes.device),
        )

    suppresses = suppression_matrix(boxes, scores, classes, threshold)
    undecided = torch.ones(count, dtype=torch.bool, device=boxes.device)
    kept = torch.zeros(count, dtype=torch.bool, device=boxes.device)

    for _ in range(rounds):
        blocked = (suppresses & undecided.unsqueeze(1)).any(dim=0)
        newly_kept = undecided & ~blocked
        kept |= newly_kept
        undecided = undecided & ~newly_kept
        victims = (suppresses & newly_kept.unsqueeze(1)).any(dim=0)
        undecided = undecided & ~victims

    converged = ~undecided.any()
    order = torch.argsort(scores, descending=True, stable=True)
    return order[kept[order]], converged


def merge_detections(
    dets,
    frame_w,
    frame_h,
    nms_iou: float,
    *,
    rounds: int = 8,
    strict: bool = True,
) -> Tuple["object", "object"]:
    """Device twin of ``chachak.boxes.merge_predictions``. Returns ``(kept, converged)``.

    ``dets`` is a compacted ``[M, 6]`` of frame-normalized Friendy rows — already concatenated
    across regions, which is what ``merge_predictions`` receives after its own ``torch.cat``.

    Like the original this has **no size floor**. It used to take a ``min_box_size`` that the
    person-crop pipeline fed ``detector.min_box_size``, which silently discarded nearly every
    real detection: that floor sizes *person crops going in* (hundreds of px) while these are
    the model's own item-level boxes coming out (a helmet is tens of px). Two different scales
    must never share a threshold. The parameter is gone rather than merely unused, deliberately.
    """
    import torch

    from . import geometry, tail

    if dets.shape[0] == 0:
        return dets, torch.ones((), dtype=torch.bool, device=dets.device)

    boxes = geometry.xywhn_to_xyxy_batched(dets[:, tail.BOX_COLUMNS], frame_w, frame_h)
    keep, converged = class_aware_overlap_nms(
        boxes,
        dets[:, tail.SCORE_COLUMN],
        dets[:, tail.CLASS_COLUMN].to(torch.int64),
        nms_iou,
        rounds=rounds,
        strict=strict,
    )
    return dets[keep], converged


def check_converged(converged, *, strict: bool, rounds: int, where: str = "NMS") -> None:
    """Raise if ``converged`` is false. **Synchronizes** — call only at a stage boundary.

    Deliberately a separate call rather than something :func:`class_aware_overlap_nms` does
    itself: reading the flag is a host synchronization, and only the caller knows where one is
    already being paid. Silently leaving boxes undecided would over-suppress, which is precisely
    the regression this package must not introduce, so the default is to fail loudly.
    """
    if not strict:
        return
    if not bool(converged):
        raise RuntimeError(
            f"{where} did not converge in {rounds} rounds: a suppression chain deeper than "
            f"that is present, and the remaining boxes would be silently dropped. Raise "
            f"GpuOptions.nms_rounds (chains this deep are unusual — check for many "
            f"near-identical stacked boxes first)."
        )
