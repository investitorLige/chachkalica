"""Class list handling, shared by the webhook and sync paths.

The class index written into each `.txt` line is the 0-based position of the
label in `data/source/<dataset>/classes.txt` (YOLO convention). The COCO
`category_id` is that index + 1 (COCO ids are conventionally 1-based).
"""

from pathlib import Path


def load_class_names(classes_file: Path) -> list[str]:
    """Read class names from a classes.txt, ignoring blank and `#` lines.

    Mirrors the subset of `label-studio.py:parse_classes_file` that matters for
    export: every non-empty, non-comment line is a class name, in order.
    """
    names: list[str] = []
    for raw in Path(classes_file).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        names.append(line)
    if not names:
        raise RuntimeError(f"{classes_file} has no class names")
    return names


def name_to_index(names: list[str]) -> dict[str, int]:
    return {name: index for index, name in enumerate(names)}
