"""Build camera_v6 without changing or deleting any v5 sample.

The v5 dataset is copied byte-for-byte first. Training images containing
``paper`` or ``note`` then receive two deterministic variants:

1. horizontal flip;
2. whole-scene distance simulation (scale down with reflected borders).

Validation and test remain untouched. Every label row, including unrelated
student/phone/door boxes and segmentation polygons, is transformed together.
"""

from __future__ import annotations

import csv
import json
import shutil
from collections import Counter
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "ultimate_exam_dataset_camera_v5_door"
OUTPUT = ROOT / "ultimate_exam_dataset_camera_v6_paper_note"
PAPER_CLASS = 2
NOTE_CLASS = 3
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_label(path: Path) -> list[list[float]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        class_id = int(parts[0])
        values = [float(value) for value in parts[1:]]
        if len(values) != 4 and (len(values) < 6 or len(values) % 2):
            raise ValueError(f"Unsupported YOLO row in {path}: {line}")
        rows.append([class_id, *values])
    return rows


def write_label(path: Path, rows: list[list[float]]) -> None:
    lines = []
    for row in rows:
        class_id, values = int(row[0]), row[1:]
        lines.append(
            f"{class_id} " + " ".join(f"{value:.6f}" for value in values)
        )
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def horizontal_flip(rows: list[list[float]]) -> list[list[float]]:
    transformed = []
    for row in rows:
        class_id, values = int(row[0]), row[1:]
        if len(values) == 4:
            cx, cy, width, height = values
            transformed.append([class_id, 1.0 - cx, cy, width, height])
        else:
            points = values[:]
            for index in range(0, len(points), 2):
                points[index] = 1.0 - points[index]
            transformed.append([class_id, *points])
    return transformed


def scale_with_padding(
    rows: list[list[float]],
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
    left: int,
    top: int,
) -> list[list[float]]:
    scale_x = target_width / source_width
    scale_y = target_height / source_height
    offset_x = left / source_width
    offset_y = top / source_height
    transformed = []
    for row in rows:
        class_id, values = int(row[0]), row[1:]
        if len(values) == 4:
            cx, cy, width, height = values
            transformed.append(
                [
                    class_id,
                    offset_x + cx * scale_x,
                    offset_y + cy * scale_y,
                    width * scale_x,
                    height * scale_y,
                ]
            )
        else:
            points = values[:]
            for index in range(0, len(points), 2):
                points[index] = offset_x + points[index] * scale_x
                points[index + 1] = offset_y + points[index + 1] * scale_y
            transformed.append([class_id, *points])
    return transformed


def find_image(stem: str) -> Path:
    matches = [
        path
        for path in (OUTPUT / "images" / "train").glob(f"{stem}.*")
        if path.suffix.lower() in IMAGE_SUFFIXES
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one image for {stem}, found {matches}")
    return matches[0]


def class_counts(dataset: Path) -> dict[str, dict[str, int]]:
    result = {}
    for split in ("train", "val", "test"):
        counts = Counter()
        for label_path in (dataset / "labels" / split).glob("*.txt"):
            for row in parse_label(label_path):
                counts[int(row[0])] += 1
        result[split] = {str(class_id): counts[class_id] for class_id in range(5)}
    return result


def validate_pairs(dataset: Path) -> None:
    for split in ("train", "val", "test"):
        image_stems = {
            path.stem
            for path in (dataset / "images" / split).iterdir()
            if path.suffix.lower() in IMAGE_SUFFIXES
        }
        label_stems = {
            path.stem for path in (dataset / "labels" / split).glob("*.txt")
        }
        if image_stems != label_stems:
            raise RuntimeError(
                f"{split} image/label mismatch: "
                f"images_only={len(image_stems-label_stems)}, "
                f"labels_only={len(label_stems-image_stems)}"
            )


def main() -> None:
    if not BASE.is_dir():
        raise FileNotFoundError(BASE)
    if OUTPUT.exists():
        raise SystemExit(
            f"Output already exists: {OUTPUT}; keep it or remove it explicitly"
        )

    before = class_counts(BASE)
    shutil.copytree(BASE, OUTPUT)
    selected = []
    manifest = []

    for label_path in sorted((OUTPUT / "labels" / "train").glob("*.txt")):
        rows = parse_label(label_path)
        classes = {int(row[0]) for row in rows}
        if not classes.intersection({PAPER_CLASS, NOTE_CLASS}):
            continue
        selected.append(label_path.stem)
        image_path = find_image(label_path.stem)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Unreadable image: {image_path}")
        height, width = image.shape[:2]

        flipped_name = f"paper_note_aug_{label_path.stem}_flip.jpg"
        flipped_image = cv2.flip(image, 1)
        cv2.imwrite(
            str(OUTPUT / "images" / "train" / flipped_name),
            flipped_image,
            [cv2.IMWRITE_JPEG_QUALITY, 92],
        )
        write_label(
            OUTPUT / "labels" / "train" / Path(flipped_name).with_suffix(".txt"),
            horizontal_flip(rows),
        )
        manifest.append((flipped_name, label_path.stem, "horizontal_flip"))

        # Deterministic 72% whole-scene scale. Reflecting the image border
        # avoids a synthetic black frame while preserving exact coordinates.
        scale = 0.72
        scaled_width = max(2, round(width * scale))
        scaled_height = max(2, round(height * scale))
        resized = cv2.resize(
            image,
            (scaled_width, scaled_height),
            interpolation=cv2.INTER_AREA,
        )
        left = (width - scaled_width) // 2
        right = width - scaled_width - left
        top = (height - scaled_height) // 2
        bottom = height - scaled_height - top
        distant_image = cv2.copyMakeBorder(
            resized,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_REFLECT_101,
        )
        distant_image = cv2.GaussianBlur(distant_image, (3, 3), 0)
        distant_name = f"paper_note_aug_{label_path.stem}_distance72.jpg"
        cv2.imwrite(
            str(OUTPUT / "images" / "train" / distant_name),
            distant_image,
            [cv2.IMWRITE_JPEG_QUALITY, 90],
        )
        write_label(
            OUTPUT / "labels" / "train" / Path(distant_name).with_suffix(".txt"),
            scale_with_padding(
                rows,
                width,
                height,
                scaled_width,
                scaled_height,
                left,
                top,
            ),
        )
        manifest.append((distant_name, label_path.stem, "distance72_blur"))

    validate_pairs(OUTPUT)
    after = class_counts(OUTPUT)
    dataset_yaml = OUTPUT / "dataset.yaml"
    dataset_yaml.write_text(
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
    with (OUTPUT / "paper_note_augmentation_manifest.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["generated_file", "source_stem", "augmentation"])
        writer.writerows(manifest)
    summary = {
        "base_dataset": str(BASE),
        "preserved_existing_samples": True,
        "selected_training_images": len(selected),
        "generated_training_images": len(manifest),
        "validation_augmented": False,
        "test_augmented": False,
        "class_counts_before": before,
        "class_counts_after": after,
    }
    (OUTPUT / "paper_note_build_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
