"""Create an isolated v7 dataset by adding the reviewed solo-phone frames.

The frozen v6 dataset is copied first and is never modified.  All 100 reviewed
images go into the v7 training split only.  Empty YOLO label files are kept as
intentional no-phone negative samples.
"""

from __future__ import annotations

import csv
import json
import shutil
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "ultimate_exam_dataset_camera_v6_paper_note"
REVIEW = ROOT / "Training_Additions" / "solo_phone_hardcases_20260730_review"
OUTPUT = ROOT / "ultimate_exam_dataset_camera_v7_phone_hardcases"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAMES = ["student", "phone", "paper", "note", "door"]


def parse_label(path: Path) -> list[list[float]]:
    rows: list[list[float]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{path}:{line_number}: expected 5 YOLO fields")
        class_id = int(parts[0])
        values = [float(value) for value in parts[1:]]
        if class_id != 1:
            raise ValueError(
                f"{path}:{line_number}: reviewed set must contain only phone (1)"
            )
        cx, cy, width, height = values
        if not all(0.0 <= value <= 1.0 for value in values):
            raise ValueError(f"{path}:{line_number}: coordinate outside [0, 1]")
        if width <= 0.0 or height <= 0.0:
            raise ValueError(f"{path}:{line_number}: non-positive box size")
        rows.append([float(class_id), cx, cy, width, height])
    return rows


def find_images(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def validate_pairs(dataset: Path) -> None:
    for split in ("train", "val", "test"):
        image_stems = {path.stem for path in find_images(dataset / "images" / split)}
        label_stems = {
            path.stem for path in (dataset / "labels" / split).glob("*.txt")
        }
        if image_stems != label_stems:
            raise RuntimeError(
                f"{split} pair mismatch: "
                f"images_only={len(image_stems-label_stems)}, "
                f"labels_only={len(label_stems-image_stems)}"
            )


def dataset_counts(dataset: Path) -> dict[str, object]:
    result: dict[str, object] = {}
    for split in ("train", "val", "test"):
        class_counts: Counter[int] = Counter()
        empty_labels = 0
        labels = sorted((dataset / "labels" / split).glob("*.txt"))
        for label in labels:
            lines = [line for line in label.read_text(encoding="utf-8").splitlines()
                     if line.strip()]
            if not lines:
                empty_labels += 1
            for line in lines:
                class_counts[int(line.split()[0])] += 1
        result[split] = {
            "images": len(find_images(dataset / "images" / split)),
            "empty_labels": empty_labels,
            "instances": {
                CLASS_NAMES[class_id]: class_counts[class_id]
                for class_id in range(len(CLASS_NAMES))
            },
        }
    return result


def main() -> None:
    if not BASE.is_dir():
        raise FileNotFoundError(BASE)
    image_dir = REVIEW / "images"
    label_dir = REVIEW / "labels_working"
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError("Reviewed image/label directories are missing")
    if OUTPUT.exists():
        raise SystemExit(
            f"Output already exists: {OUTPUT}; refusing to overwrite an experiment"
        )

    images = find_images(image_dir)
    if len(images) != 100:
        raise RuntimeError(f"Expected exactly 100 reviewed images, found {len(images)}")

    reviewed: list[tuple[Path, Path, list[list[float]]]] = []
    for image in images:
        label = label_dir / f"{image.stem}.txt"
        if not label.is_file():
            raise FileNotFoundError(label)
        reviewed.append((image, label, parse_label(label)))

    positive_images = sum(bool(rows) for _, _, rows in reviewed)
    empty_images = len(reviewed) - positive_images
    phone_boxes = sum(len(rows) for _, _, rows in reviewed)
    if (positive_images, empty_images, phone_boxes) != (87, 13, 93):
        raise RuntimeError(
            "Reviewed-set fingerprint changed; expected "
            "87 positives, 13 negatives and 93 phone boxes, got "
            f"{positive_images}, {empty_images}, {phone_boxes}"
        )

    before = dataset_counts(BASE)
    shutil.copytree(BASE, OUTPUT)

    manifest_rows: list[dict[str, object]] = []
    for index, (image, label, rows) in enumerate(reviewed, start=1):
        destination_stem = f"solo_phone_review_{index:03d}_{image.stem}"
        destination_image = (
            OUTPUT / "images" / "train" / f"{destination_stem}{image.suffix.lower()}"
        )
        destination_label = OUTPUT / "labels" / "train" / f"{destination_stem}.txt"
        shutil.copy2(image, destination_image)
        shutil.copy2(label, destination_label)
        manifest_rows.append(
            {
                "new_stem": destination_stem,
                "source_image": str(image.relative_to(ROOT)),
                "phone_boxes": len(rows),
                "sample_role": "phone_positive" if rows else "phone_hard_negative",
            }
        )

    (OUTPUT / "dataset.yaml").write_text(
        "\n".join(
            [
                f"path: {OUTPUT.as_posix()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "nc: 5",
                "names: ['student', 'phone', 'paper', 'note', 'door']",
                "",
            ]
        ),
        encoding="utf-8",
    )

    with (OUTPUT / "phone_hardcases_manifest.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    validate_pairs(OUTPUT)
    after = dataset_counts(OUTPUT)
    summary = {
        "base_dataset": str(BASE),
        "output_dataset": str(OUTPUT),
        "review_source": str(REVIEW),
        "reviewed_images_added_to_train_only": len(reviewed),
        "reviewed_positive_images": positive_images,
        "reviewed_empty_negative_images": empty_images,
        "reviewed_phone_boxes": phone_boxes,
        "before": before,
        "after": after,
        "validation_and_test_unchanged": True,
    }
    (OUTPUT / "build_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
