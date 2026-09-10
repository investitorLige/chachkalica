"""Detect (and optionally prune) images duplicated across two or more datasets.

Every image is fingerprinted twice: an exact MD5 of its file bytes (catches
byte-identical copies) and an 8x8 difference hash (catches the same photo
re-exported at a different size or compression — the common case when a
dataset has passed through Roboflow/Kaggle more than once, where an MD5 match
never happens but the picture is still the same). MD5 is tried first since
it's a stronger signal; the perceptual hash is only consulted when no exact
copy exists in the other dataset, and it matches within a few differing bits
rather than exactly — re-encoding flips some of the gradient comparisons, so
demanding an identical hash would miss most of the pairs this is here for.
"""

import hashlib
import shutil
from datetime import datetime
from pathlib import Path

import cv2

from fleet.models import Dataset
from fleet.reconcile import writer
from fleet.services import datasets as datasets_svc
from fleet.services import lsapi
from fleet.services.paths import source_root

_HASH_SIZE = 8
_HASH_BITS = _HASH_SIZE * _HASH_SIZE
# Re-encoding the same photo flips a few gradient comparisons, so near-duplicate
# matching allows this many differing bits out of 64. The hash is indexed in
# bands to keep the join a dict lookup: two hashes within _DHASH_MAX_DISTANCE
# bits can spoil at most that many of the _DHASH_BANDS bands, so they are
# guaranteed to still agree on at least one whole band. Recall stays exact only
# while _DHASH_MAX_DISTANCE < _DHASH_BANDS.
_DHASH_BANDS = 8
_DHASH_BAND_BITS = _HASH_BITS // _DHASH_BANDS
_DHASH_MAX_DISTANCE = 5
_PRUNE_BACKUP_DIR = ".overlap_prune_backups"


def _md5_of_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dhash(path: Path, hash_size: int = _HASH_SIZE) -> int | None:
    """Gradient hash: robust to resizing/re-encoding, unlike a byte hash."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    small = cv2.resize(image, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    bits = 0
    for row in range(hash_size):
        for col in range(hash_size):
            bits = (bits << 1) | int(small[row, col] > small[row, col + 1])
    return bits


def _dhash_bands(dhash: int) -> list[tuple[int, int]]:
    """The (band index, band value) keys a hash is indexed and looked up under."""
    mask = (1 << _DHASH_BAND_BITS) - 1
    return [
        (band, (dhash >> (band * _DHASH_BAND_BITS)) & mask)
        for band in range(_DHASH_BANDS)
    ]


def _near_hits(dhash: int, by_band: dict[tuple[int, int], list[dict]]) -> list[dict]:
    """Fingerprints within _DHASH_MAX_DISTANCE bits of ``dhash``, each returned once.

    Candidates are identified by their ``key`` when they carry one and by their
    path otherwise: :func:`group_overlaps` joins several datasets at once, where
    two entries can share a path (two Dataset rows are allowed the same name,
    hence the same directory) and must still be treated as separate copies.
    """
    hits = []
    seen: set = set()
    for band_key in _dhash_bands(dhash):
        for candidate in by_band.get(band_key, ()):
            identity = candidate.get("key", candidate["path"])
            if identity in seen:
                continue
            seen.add(identity)
            if (candidate["dhash"] ^ dhash).bit_count() <= _DHASH_MAX_DISTANCE:
                hits.append(candidate)
    return hits


def fingerprint_dataset(dataset: Dataset) -> list[dict]:
    """MD5 + perceptual hash for every image in a dataset, ready for joining."""
    dataset_dir = source_root() / dataset.name
    prints = []
    for image in lsapi.list_dataset_images(dataset_dir):
        try:
            md5 = _md5_of_file(image)
        except OSError:
            continue
        prints.append({"path": image, "md5": md5, "dhash": _dhash(image)})
    return prints


def compare_pair(left: Dataset, right: Dataset, left_prints=None, right_prints=None) -> dict:
    """Overlap report between two datasets: exact (md5) + near-duplicate (dhash) matches.

    ``left_prints``/``right_prints`` let :func:`find_overlaps` reuse fingerprints
    already computed once per dataset instead of re-hashing for every pair.
    """
    left_prints = fingerprint_dataset(left) if left_prints is None else left_prints
    right_prints = fingerprint_dataset(right) if right_prints is None else right_prints

    by_md5: dict[str, list[dict]] = {}
    by_band: dict[tuple[int, int], list[dict]] = {}
    for fp in right_prints:
        by_md5.setdefault(fp["md5"], []).append(fp)
        if fp["dhash"] is not None:
            for key in _dhash_bands(fp["dhash"]):
                by_band.setdefault(key, []).append(fp)

    matches = []
    for fp in left_prints:
        hits = by_md5.get(fp["md5"])
        kind = "exact"
        if not hits and fp["dhash"] is not None:
            hits = _near_hits(fp["dhash"], by_band)
            kind = "near"
        if hits:
            matches.extend({"left": fp["path"], "right": hit["path"], "kind": kind} for hit in hits)

    return {
        "left": left,
        "right": right,
        "left_count": len(left_prints),
        "right_count": len(right_prints),
        "matches": matches,
        "exact_count": sum(1 for m in matches if m["kind"] == "exact"),
        "near_count": sum(1 for m in matches if m["kind"] == "near"),
    }


def find_overlaps(datasets: list[Dataset], prints: dict[int, list[dict]] | None = None) -> list[dict]:
    """Every pairwise overlap report among the given datasets (order-independent).

    ``prints`` lets a caller that needs more than one view of the same datasets
    (the admin report renders both this pair table and :func:`group_overlaps`)
    hand in fingerprints it already has. Hashing dominates the cost by orders of
    magnitude — see ``jobs.PRUNE_INTRA_DUPLICATES_JOB_TIMEOUT`` — so computing
    it twice for one page is not an option.
    """
    if prints is None:
        prints = {dataset.pk: fingerprint_dataset(dataset) for dataset in datasets}
    reports = []
    for i, left in enumerate(datasets):
        for right in datasets[i + 1:]:
            reports.append(compare_pair(left, right, prints[left.pk], prints[right.pk]))
    return reports


def _union_find(keys):
    """Tiny union-find over hashable keys: returns ``(find, union)``."""
    parent = {key: key for key in keys}

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    return find, union


def group_overlaps(datasets: list[Dataset], prints: dict[int, list[dict]] | None = None) -> dict:
    """Cross-dataset duplicates as one group per picture, listing every copy of it.

    Where :func:`find_overlaps` answers "which images match, pair of datasets by
    pair of datasets", this answers "here is one picture and every copy of it
    that exists" — the shape a prune UI needs. A pair list can't drive one: a
    near-duplicate legitimately matches several images on the other side, so the
    same picture appears in several rows and per-row keep/delete choices can
    contradict each other. A group has exactly one survivor by construction.

    Only edges *between* datasets are unioned, so every group spans at least two
    of them — duplicates inside a single dataset are ``find_intra_duplicates``'
    job. A second copy sitting in the same dataset is still pulled in when it
    also matches across, since it is the same picture and pruning around it
    would leave the overlap in place.
    """
    if prints is None:
        prints = {dataset.pk: fingerprint_dataset(dataset) for dataset in datasets}

    order = {dataset.pk: index for index, dataset in enumerate(datasets)}
    entries = []
    for dataset in datasets:
        for fp in prints.get(dataset.pk, ()):
            entries.append({**fp, "dataset": dataset, "key": (dataset.pk, fp["path"])})

    find, union = _union_find(entry["key"] for entry in entries)

    # Exact edges: an MD5 bucket holding more than one dataset is the same bytes
    # in each of them, so every copy in the bucket belongs to one group.
    by_md5: dict[str, list[dict]] = {}
    for entry in entries:
        by_md5.setdefault(entry["md5"], []).append(entry)
    for bucket in by_md5.values():
        if len({entry["dataset"].pk for entry in bucket}) < 2:
            continue
        for entry in bucket[1:]:
            union(bucket[0]["key"], entry["key"])

    # Near edges: the same band index as compare_pair, except joined over every
    # dataset at once and only where the two sides come from different ones.
    by_band: dict[tuple[int, int], list[dict]] = {}
    for entry in entries:
        if entry["dhash"] is None:
            continue
        for band_key in _dhash_bands(entry["dhash"]):
            by_band.setdefault(band_key, []).append(entry)
    for entry in entries:
        if entry["dhash"] is None:
            continue
        for hit in _near_hits(entry["dhash"], by_band):
            if hit["dataset"].pk != entry["dataset"].pk:
                union(entry["key"], hit["key"])

    clusters: dict[tuple, list[dict]] = {}
    for entry in entries:
        clusters.setdefault(find(entry["key"]), []).append(entry)

    groups = []
    for members in clusters.values():
        if len(members) < 2:
            continue
        members.sort(key=lambda entry: (order[entry["dataset"].pk], entry["path"].name))
        groups.append({
            # "exact" only when every copy is byte-identical; one re-encoded
            # copy in the group makes the whole thing worth a second look.
            "kind": "exact" if len({entry["md5"] for entry in members}) == 1 else "near",
            "items": [
                {
                    "dataset": entry["dataset"],
                    "path": entry["path"],
                    "name": entry["path"].name,
                    # What the prune form posts back. Filenames are unique within
                    # a dataset (list_dataset_images does not recurse), so the pk
                    # plus the name identifies a copy without a traversable path
                    # ever reaching the browser.
                    "token": f"{entry['dataset'].pk}:{entry['path'].name}",
                }
                for entry in members
            ],
        })
    groups.sort(key=lambda group: (order[group["items"][0]["dataset"].pk],
                                   group["items"][0]["name"]))

    per_dataset = []
    for dataset in datasets:
        copies = sum(1 for group in groups
                     for item in group["items"] if item["dataset"].pk == dataset.pk)
        per_dataset.append({
            "dataset": dataset,
            "image_count": len(prints.get(dataset.pk, ())),
            "overlapping": copies,
        })

    return {
        "datasets": datasets,
        "per_dataset": per_dataset,
        "groups": groups,
        "group_count": len(groups),
        "copy_count": sum(len(group["items"]) for group in groups),
        # One survivor per group, so this is what a full prune would delete.
        "extra_count": sum(len(group["items"]) - 1 for group in groups),
        "exact_group_count": sum(1 for group in groups if group["kind"] == "exact"),
        "near_group_count": sum(1 for group in groups if group["kind"] == "near"),
    }


def _cluster_by_dhash(prints: list[dict]) -> list[list[dict]]:
    """Group fingerprints into near-duplicate clusters via union-find over dhash bands.

    Unlike ``compare_pair``, this is a self-join: every image is only compared
    against the others in the same list, so groups (not left/right pairs) are
    the natural output.
    """
    find, union = _union_find(fp["path"] for fp in prints)

    by_band: dict[tuple[int, int], list[dict]] = {}
    for fp in prints:
        for key in _dhash_bands(fp["dhash"]):
            by_band.setdefault(key, []).append(fp)

    for fp in prints:
        for hit in _near_hits(fp["dhash"], by_band):
            if hit["path"] != fp["path"]:
                union(fp["path"], hit["path"])

    clusters: dict[Path, list[dict]] = {}
    for fp in prints:
        clusters.setdefault(find(fp["path"]), []).append(fp)
    return [
        sorted(group, key=lambda fp: fp["path"].name)
        for group in clusters.values() if len(group) > 1
    ]


def find_intra_duplicates(dataset: Dataset) -> dict:
    """Duplicate/near-duplicate images within a single dataset.

    Groups by exact MD5 first (byte-identical copies), then clusters whatever
    is left by dhash (re-exported/re-compressed copies of the same photo) —
    an image already accounted for in an exact-match group is excluded from
    the dhash pass so it isn't counted twice. Each group's alphabetically
    first path is treated as the keeper; the rest are reported as prunable.
    """
    prints = fingerprint_dataset(dataset)

    by_md5: dict[str, list[dict]] = {}
    for fp in prints:
        by_md5.setdefault(fp["md5"], []).append(fp)
    exact_groups = [
        sorted(group, key=lambda fp: fp["path"].name)
        for group in by_md5.values() if len(group) > 1
    ]
    exact_paths = {fp["path"] for group in exact_groups for fp in group}

    remaining = [fp for fp in prints if fp["path"] not in exact_paths and fp["dhash"] is not None]
    near_groups = _cluster_by_dhash(remaining)

    def prunable(groups: list[list[dict]]) -> list[Path]:
        return [fp["path"] for group in groups for fp in group[1:]]

    return {
        "dataset": dataset,
        "image_count": len(prints),
        "exact_groups": exact_groups,
        "near_groups": near_groups,
        "exact_duplicate_extra": sum(len(g) - 1 for g in exact_groups),
        "near_duplicate_extra": sum(len(g) - 1 for g in near_groups),
        "prunable_exact": prunable(exact_groups),
        "prunable_near": prunable(near_groups),
    }


def prune_intra_duplicates(dataset: Dataset) -> dict:
    """Delete extra copies within one dataset's duplicate/near-duplicate clusters.

    Always re-fingerprints rather than trusting a report computed earlier,
    since the picture on disk is the only thing safe to prune from. (The
    cross-dataset prune is the deliberate exception — there the operator picked
    individual copies, so re-deriving the set would discard their decision;
    see :func:`prune_paths`.) Keeps one image per cluster; every deletion is backed
    up first.
    """
    report = find_intra_duplicates(dataset)
    result = {"deleted_images": 0, "deleted_labels": 0, "backup_dir": ""}
    to_prune = report["prunable_exact"] + report["prunable_near"]
    if not to_prune:
        return result

    backup_root = source_root() / _PRUNE_BACKUP_DIR / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    seen: set[Path] = set()
    for path in to_prune:
        if path not in seen and path.exists():
            _backup_and_delete(path, dataset, backup_root, result)
            seen.add(path)

    if result["deleted_images"] or result["deleted_labels"]:
        result["backup_dir"] = str(backup_root)
    return result


def _backup_one(path: Path, dataset_dir: Path, backup_root: Path, dataset_name: str) -> None:
    try:
        relative = path.relative_to(dataset_dir)
    except ValueError:
        relative = Path(path.name)
    target = backup_root / dataset_name / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, target)


def _backup_and_delete(image_path: Path, dataset: Dataset, backup_root: Path, result: dict) -> None:
    dataset_dir = source_root() / dataset.name
    _backup_one(image_path, dataset_dir, backup_root, dataset.name)
    if writer.delete(image_path):
        result["deleted_images"] += 1

    label_path = datasets_svc.labels_source_dir(dataset) / f"{image_path.stem}.txt"
    if label_path.exists():
        _backup_one(label_path, dataset_dir, backup_root, dataset.name)
        if writer.delete(label_path):
            result["deleted_labels"] += 1


def prune_paths(selection: list[tuple[Dataset, list[str]]]) -> dict:
    """Delete named images (and their label files) from specific datasets.

    Takes the operator's per-copy decision from the overlap report rather than a
    match report, so unlike the detection passes there is nothing to re-derive
    here: the chosen names *are* the input. Anything already gone is counted as
    skipped instead of failing the batch, since a report can be minutes old by
    the time a worker picks the job up.

    Every name is resolved inside the dataset's own image directory and rejected
    unless it names a file already there, so a hand-edited form post cannot
    reach a path outside the dataset.
    """
    result = {"deleted_images": 0, "deleted_labels": 0, "backup_dir": "", "skipped": 0}
    backup_root = source_root() / _PRUNE_BACKUP_DIR / datetime.now().strftime("%Y%m%d-%H%M%S-%f")

    for dataset, names in selection:
        image_dir = lsapi.image_source_dir(source_root() / dataset.name)
        for name in dict.fromkeys(names):  # de-duplicated, order preserved
            path = resolve_dataset_image(image_dir, name)
            if path is None:
                result["skipped"] += 1
                continue
            _backup_and_delete(path, dataset, backup_root, result)

    if result["deleted_images"] or result["deleted_labels"]:
        result["backup_dir"] = str(backup_root)
    return result


def resolve_dataset_image(image_dir: Path, name: str) -> Path | None:
    """``image_dir/name`` when that names an existing image file, else None.

    Rejects anything with a path separator in it up front, so ``..`` and
    absolute paths can never resolve — the check is on the *name*, not on the
    joined path, because ``Path("/a") / "/etc/passwd"`` is ``/etc/passwd``.
    """
    if not name or "/" in name or "\\" in name or name in (".", ".."):
        return None
    candidate = image_dir / name
    if candidate.suffix.lower() not in lsapi.IMAGE_EXTENSIONS or not candidate.is_file():
        return None
    return candidate
