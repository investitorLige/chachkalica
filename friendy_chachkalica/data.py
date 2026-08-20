from pathlib import Path
from typing import Optional

from torch.utils.data import DataLoader

try:
    from .config import DatasetConfig, ExperimentConfig
    from .formats import xywhn_to_xyxy
except ImportError:
    from config import DatasetConfig, ExperimentConfig
    from formats import xywhn_to_xyxy


IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def _resolve_image_dir(images_root):
    images_root = Path(images_root).resolve()
    if not images_root.is_dir():
        raise FileNotFoundError(f"Could not resolve images directory: {images_root}")
    return sorted(
        path.resolve()
        for path in images_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _image_to_label_path(image_path, images_root, labels_root):
    relative_path = image_path.relative_to(images_root)
    # Two label-naming conventions occur in practice: "<stem>.txt" (drop the
    # image suffix, e.g. frame_000.txt) and "<image filename>.txt" (append,
    # e.g. frame_000.jpg.txt). Prefer the appended form when it exists on disk,
    # otherwise fall back to the stripped form (also the not-found default, so
    # callers that only test .exists() behave as before).
    stripped = (labels_root / relative_path).with_suffix(".txt")
    appended = labels_root / relative_path.parent / (relative_path.name + ".txt")
    if appended.exists() and not stripped.exists():
        return appended
    return stripped


def _read_yolo_label_file(label_path, image_width, image_height):
    boxes = []
    labels = []

    if not label_path.exists():
        return boxes, labels

    with open(label_path) as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            class_id = int(float(parts[0]))
            values = [float(value) for value in parts[1:]]
            if len(values) == 4:
                box = xywhn_to_xyxy(values, image_width, image_height)
            elif len(values) == 5:
                # YOLO OBB format (cx cy w h angle) — treat as axis-aligned bbox
                box = xywhn_to_xyxy(values[:4], image_width, image_height)
            elif len(values) >= 6 and len(values) % 2 == 0:
                box = _normalized_polygon_to_xyxy(values, image_width, image_height)
            else:
                continue

            if box[2] <= box[0] or box[3] <= box[1]:
                continue

            boxes.append(box)
            labels.append(class_id)

    return boxes, labels


def _scan_label_file(label_path, valid_class_ids):
    """Count label lines with out-of-range class ids or clearly unnormalized coords.

    Mirrors ``_read_yolo_label_file``'s parsing so it only inspects lines that
    would actually produce a box. Coordinates are flagged when far outside
    [0, 1] (i.e. pixel coordinates in a file that must be normalized); for the
    5-value OBB form the trailing angle is exempt from the range check.
    """
    bad_class_lines = 0
    unnormalized_lines = 0
    valid_class_ids = set(valid_class_ids)
    with open(label_path) as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                class_id = int(float(parts[0]))
                values = [float(value) for value in parts[1:]]
            except ValueError:
                continue
            if class_id not in valid_class_ids:
                bad_class_lines += 1
            coords = values[:4] if len(values) == 5 else values
            if any(value < -0.5 or value > 1.5 for value in coords):
                unnormalized_lines += 1
    return bad_class_lines, unnormalized_lines


def _normalized_polygon_to_xyxy(points, image_width, image_height):
    xs = points[0::2]
    ys = points[1::2]
    x1 = min(xs) * image_width
    y1 = min(ys) * image_height
    x2 = max(xs) * image_width
    y2 = max(ys) * image_height
    return [x1, y1, x2, y2]


class YoloDetectionDataset:
    def __init__(self, images_dir, labels_dir, classes, transforms=None, require_labels=False):
        self.transforms = transforms
        self.images_root = Path(images_dir).resolve()
        self.labels_root = Path(labels_dir).resolve() if labels_dir is not None else None
        self.names = dict(classes)
        self.image_paths = _resolve_image_dir(self.images_root)
        # Corrupt/truncated files show up in real-world datasets often enough
        # that one bad JPEG shouldn't take down a whole multi-epoch run; see
        # __getitem__. Logged once per path so a repeat-offender doesn't spam
        # every epoch.
        self._warned_bad_paths = set()

        if self.labels_root is not None and not self.labels_root.is_dir():
            raise FileNotFoundError(f"Could not resolve labels directory: {self.labels_root}")

        label_count = 0
        bad_class_lines = 0
        unnormalized_lines = 0
        for image_path in self.image_paths:
            if self.labels_root is None:
                continue
            label_path = _image_to_label_path(image_path, self.images_root, self.labels_root)
            if not label_path.exists():
                continue
            label_count += 1
            bad_classes, unnormalized = _scan_label_file(label_path, self.names)
            bad_class_lines += bad_classes
            unnormalized_lines += unnormalized
        print(
            f"[data] Dataset ready: images={len(self.image_paths)} labels_found={label_count} "
            f"classes={len(self.names)} images_root={self.images_root} labels_root={self.labels_root}"
        )
        # Both anomalies below corrupt training/eval without crashing, so call
        # them out once per dataset instead of failing silently per line.
        if bad_class_lines:
            print(
                f"[data] WARNING: {bad_class_lines} label line(s) use class ids outside the "
                f"{len(self.names)} configured classes. Those boxes will crash training or be "
                f"silently dropped from eval — check that the class list matches how the "
                f"labels were exported (e.g. 0-based vs 1-based ids)."
            )
        if unnormalized_lines:
            print(
                f"[data] WARNING: {unnormalized_lines} label line(s) have coordinates far "
                f"outside [0, 1]. Labels must be normalized YOLO xywh; pixel-coordinate "
                f"labels produce garbage boxes without crashing."
            )
        # Zero labels found almost always means a path/naming mismatch, not a
        # dataset that is genuinely all-background. For eval sets this silently
        # produces meaningless loss/mAP (every prediction scored against empty
        # ground truth), so fail loud. Train tolerates it with a warning.
        if label_count == 0 and self.labels_root is not None:
            message = (
                f"No labels matched any of {len(self.image_paths)} images under "
                f"{self.labels_root} (checked both '<stem>.txt' and "
                f"'<image>.txt' naming). Check the labels directory and filename "
                f"convention."
            )
            if require_labels:
                raise FileNotFoundError(message)
            print(f"[data] WARNING: {message}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index, _attempt=0):
        import numpy as np
        import torch
        from PIL import Image, ImageOps

        image_path = self.image_paths[index]
        try:
            # Match browser/display orientation before deriving dimensions and boxes.
            image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
        except OSError as exc:
            # Truncated/corrupt files happen in real datasets (bad export,
            # interrupted copy, ...). Crashing the whole DataLoader worker over
            # one file kills a run that may be hours into training, so skip it
            # and substitute a neighboring sample instead. Bounded by dataset
            # size so a dataset that's corrupt end-to-end still raises instead
            # of recursing forever.
            if image_path not in self._warned_bad_paths:
                self._warned_bad_paths.add(image_path)
                print(f"[data] WARNING: skipping unreadable image {image_path}: {exc}")
            if _attempt >= len(self):
                raise
            return self.__getitem__((index + 1) % len(self), _attempt=_attempt + 1)
        width, height = image.size
        label_path = (_image_to_label_path(image_path, self.images_root, self.labels_root)
                      if self.labels_root is not None else None)
        boxes, labels = (_read_yolo_label_file(label_path, width, height)
                         if label_path is not None else ([], []))

        image = np.asarray(image).copy()
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels_tensor = torch.tensor(labels, dtype=torch.int64)
        area = (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (boxes_tensor[:, 3] - boxes_tensor[:, 1])

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([index], dtype=torch.int64),
            "area": area,
            "iscrowd": torch.zeros((len(labels),), dtype=torch.int64),
            "image_path": str(image_path),
            "label_path": str(label_path) if label_path is not None else None,
            "orig_size": torch.tensor([height, width], dtype=torch.int64),
        }

        if self.transforms is not None:
            image_tensor, target = self.transforms(image_tensor, target)

        return image_tensor, target


def detection_collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


class TrainAugmentations:
    """Random train-time augmentations, each applied to a fraction of samples.

    ``fractions`` maps augmentation name -> probability per sample per epoch
    (see ``config.AUGMENTATION_KEYS``). Boxes stay absolute xyxy on the
    (possibly transformed) image; images keep their original size, so nothing
    downstream of the dataset changes shape.
    """

    SCALE_RANGE = (0.6, 1.0)  # crop window size as a fraction of each image side
    MIN_BOX_VISIBILITY = 0.25  # drop boxes with less than this area left in the crop
    MIN_BOX_SIDE_PX = 2.0  # drop boxes thinner than this after the crop resize

    def __init__(self, fractions: dict):
        self.hflip = float(fractions.get("hflip", 0.0))
        self.scale_crop = float(fractions.get("scale_crop", 0.0))

    def __call__(self, image, target):
        import torch

        # torch.rand honors the per-worker seeding the DataLoader sets up, so
        # runs stay reproducible under training.seed.
        if self.hflip > 0 and torch.rand(1).item() < self.hflip:
            image, target = _horizontal_flip(image, target)
        if self.scale_crop > 0 and torch.rand(1).item() < self.scale_crop:
            image, target = _random_scale_crop(
                image, target, self.SCALE_RANGE, self.MIN_BOX_VISIBILITY, self.MIN_BOX_SIDE_PX
            )
        return image, target


def _horizontal_flip(image, target):
    import torch

    image = torch.flip(image, dims=[2])  # (C, H, W) -> flip width
    boxes = target["boxes"]
    if boxes.numel():
        width = image.shape[2]
        flipped = boxes.clone()
        flipped[:, 0] = width - boxes[:, 2]
        flipped[:, 2] = width - boxes[:, 0]
        target["boxes"] = flipped
    return image, target


def _random_scale_crop(image, target, scale_range, min_visibility, min_side_px):
    """Crop a random window of scale_range x the image, resize back to full size.

    Boxes are shifted into the crop, clipped at its edges, rescaled with the
    image, and dropped when the crop leaves too little of them (less than
    ``min_visibility`` of their area or a side under ``min_side_px``).
    """
    import torch

    _, height, width = image.shape
    low, high = scale_range
    scale = low + (high - low) * torch.rand(1).item()
    crop_h = max(1, min(height, round(height * scale)))
    crop_w = max(1, min(width, round(width * scale)))
    if crop_h == height and crop_w == width:
        return image, target

    top = int(torch.randint(0, height - crop_h + 1, (1,)).item())
    left = int(torch.randint(0, width - crop_w + 1, (1,)).item())

    cropped = image[:, top : top + crop_h, left : left + crop_w]
    image = torch.nn.functional.interpolate(
        cropped.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False
    ).squeeze(0)

    boxes = target["boxes"]
    if boxes.numel():
        shifted = boxes - boxes.new_tensor([left, top, left, top])
        clipped = shifted.clone()
        clipped[:, 0::2] = clipped[:, 0::2].clamp(0, crop_w)
        clipped[:, 1::2] = clipped[:, 1::2].clamp(0, crop_h)

        # Rescale from crop coordinates back to the full-size image.
        scale_x = width / crop_w
        scale_y = height / crop_h
        clipped = clipped * clipped.new_tensor([scale_x, scale_y, scale_x, scale_y])

        widths = clipped[:, 2] - clipped[:, 0]
        heights = clipped[:, 3] - clipped[:, 1]
        original_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        visible_area = widths * heights / (scale_x * scale_y)
        keep = (
            (widths >= min_side_px)
            & (heights >= min_side_px)
            & (visible_area >= min_visibility * original_area.clamp(min=1e-6))
        )

        target["boxes"] = clipped[keep]
        target["labels"] = target["labels"][keep]
        target["iscrowd"] = target["iscrowd"][keep]
        target["area"] = (
            (target["boxes"][:, 2] - target["boxes"][:, 0])
            * (target["boxes"][:, 3] - target["boxes"][:, 1])
        )
    return image, target


def build_train_augmentations(dataset_config: DatasetConfig):
    """The transforms callable for a train dataset, or None when unconfigured."""
    fractions = {k: v for k, v in (dataset_config.augmentation or {}).items() if v > 0}
    if not fractions:
        return None
    return TrainAugmentations(fractions)


def build_dataset(
    dataset_config: DatasetConfig,
    transforms=None,
    require_labels: bool = False,
) -> YoloDetectionDataset:
    print(
        f"[data] Building dataset name={dataset_config.name} role={dataset_config.role} "
        f"classes={len(dataset_config.classes)} transforms={transforms is not None}"
    )
    return YoloDetectionDataset(
        images_dir=dataset_config.images,
        labels_dir=dataset_config.labels,
        classes=dataset_config.classes,
        transforms=transforms,
        require_labels=require_labels,
    )


def build_train_dataloader(
    config: ExperimentConfig,
    dataset_config: DatasetConfig,
) -> DataLoader:
    transforms = build_train_augmentations(dataset_config)
    if transforms is not None:
        print(
            f"[data] Train augmentations for {dataset_config.name}: "
            f"hflip={transforms.hflip} scale_crop={transforms.scale_crop}"
        )
    dataset = build_dataset(dataset_config, transforms=transforms)
    print(
        f"[data] Building train dataloader dataset={dataset_config.name} "
        f"batch_size={config.training.batch_size} workers={config.training.num_workers} "
        f"shuffle=True"
    )
    return DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        collate_fn=detection_collate_fn,
        pin_memory=_cuda_is_available(),
    )


def build_eval_dataloader(
    dataset_config: Optional[DatasetConfig],
    config: ExperimentConfig,
) -> Optional[DataLoader]:
    if dataset_config is None:
        return None

    batch_size = config.evaluation.batch_size or config.training.batch_size
    num_workers = config.evaluation.num_workers
    if num_workers is None:
        num_workers = config.training.num_workers

    # Eval requires labels: a val/test set with none makes every loss and mAP
    # meaningless, so surface it as a hard error instead of "training complete".
    dataset = build_dataset(dataset_config, require_labels=dataset_config.labels is not None)
    print(
        f"[data] Building eval dataloader dataset={dataset_config.name} "
        f"batch_size={batch_size} workers={num_workers} shuffle=False"
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=detection_collate_fn,
        pin_memory=_cuda_is_available(),
    )


def _cuda_is_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()
