"""Build an isolated five-class dataset with a real ``door`` class.

The existing camera_v2 dataset and saved baseline weights are never modified.
Door images and their pixel masks come from DoorDetect-Class-Dataset:
https://github.com/gasparramoa/DoorDetect-Class-Dataset

Class order in the new dataset:
    0 student, 1 phone, 2 paper, 3 note, 4 door

Only the training portion is augmented. Validation and test images remain
untouched so their metrics stay meaningful.
"""

from __future__ import annotations

import csv
import json
import random
import shutil
import time
import urllib.request
from pathlib import Path

import cv2
import gdown
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
BASE_DATASET = ROOT / "ultimate_exam_dataset_camera_v2"
OUTPUT_DATASET = ROOT / "ultimate_exam_dataset_camera_v5_door"
WORK_DIR = ROOT / "Training_Additions" / "door_class_v1"
SOURCE_DIR = WORK_DIR / "source"

FOLDER_IDS = {
    "train_images": "1len_1mZtyD8NnJebq9Q9qe7CQouneywu",
    "train_masks": "1hdQKDKc_MM3BtI_XkFcw7VOpow5N2cO9",
    "test_images": "1uQY0tr-VRomUjbxLQN1FvalKplYW2DSU",
    "test_masks": "1p-uIwW1fUG22-fgenp1RdYyXaH2drFcl",
}

DOOR_CLASS_ID = 4
RANDOM_SEED = 20260729


def download_folder(folder_id: str, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    entries = gdown.download_folder(
        id=folder_id,
        output=str(destination),
        quiet=True,
        use_cookies=False,
        skip_download=True,
    )
    if not entries:
        raise RuntimeError(f"Google Drive folder download failed: {folder_id}")

    for index, entry in enumerate(entries, start=1):
        filename = Path(entry.path).name
        final_path = destination / filename
        if final_path.exists() and final_path.stat().st_size > 0:
            try:
                with Image.open(final_path) as image:
                    image.verify()
                continue
            except Exception:
                final_path.unlink()

        url = (
            "https://drive.usercontent.google.com/download"
            f"?id={entry.id}&export=download&confirm=t"
        )
        temporary_path = final_path.with_suffix(final_path.suffix + ".part")
        last_error: Exception | None = None
        for attempt in range(6):
            try:
                request = urllib.request.Request(
                    url,
                    headers={"User-Agent": "camera-door-dataset-builder/1.0"},
                )
                with urllib.request.urlopen(request, timeout=90) as response:
                    temporary_path.write_bytes(response.read())
                with Image.open(temporary_path) as image:
                    image.verify()
                temporary_path.replace(final_path)
                last_error = None
                break
            except Exception as error:
                last_error = error
                if temporary_path.exists():
                    temporary_path.unlink()
                time.sleep(min(20, 2 ** attempt))
        if last_error is not None:
            raise RuntimeError(
                f"Failed to download {filename} after retries"
            ) from last_error
        if index % 20 == 0 or index == len(entries):
            print(f"  {destination.name}: {index}/{len(entries)}")
        time.sleep(0.08)


def source_files(directory: Path) -> dict[str, Path]:
    files = {}
    for path in directory.glob("*.png"):
        files[path.stem] = path
    return files


def mask_to_yolo(mask_path: Path) -> tuple[float, float, float, float]:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise RuntimeError(f"Unreadable mask: {mask_path}")
    if mask.ndim == 3:
        mask = mask[:, :, 0]

    # The source documentation defines 1 as door/doorframe and 2 as
    # background. Do not infer a box from the whole image if the mask is bad.
    ys, xs = np.where(mask == 1)
    if len(xs) == 0:
        values, counts = np.unique(mask, return_counts=True)
        raise RuntimeError(
            f"No door pixels (value 1) in {mask_path.name}; "
            f"mask values={dict(zip(values.tolist(), counts.tolist()))}"
        )

    height, width = mask.shape[:2]
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    box_w, box_h = x2 - x1, y2 - y1
    if box_w < 8 or box_h < 8:
        raise RuntimeError(f"Implausibly small door box in {mask_path.name}")

    return (
        ((x1 + x2) / 2) / width,
        ((y1 + y2) / 2) / height,
        box_w / width,
        box_h / height,
    )


def write_label(path: Path, box: tuple[float, float, float, float]) -> None:
    path.write_text(
        f"{DOOR_CLASS_ID} " + " ".join(f"{value:.6f}" for value in box) + "\n",
        encoding="utf-8",
    )


def add_original(
    image_path: Path,
    mask_path: Path,
    split: str,
    source_split: str,
    rows: list[dict[str, str]],
) -> tuple[Path, tuple[float, float, float, float]]:
    box = mask_to_yolo(mask_path)
    name = f"door_{source_split}_{image_path.stem}.png"
    destination = OUTPUT_DATASET / "images" / split / name
    shutil.copy2(image_path, destination)
    write_label(OUTPUT_DATASET / "labels" / split / f"{Path(name).stem}.txt", box)
    rows.append(
        {
            "file": name,
            "split": split,
            "source_split": source_split,
            "augmentation": "original",
            "source_image": image_path.name,
        }
    )
    return destination, box


def add_training_augmentations(
    original_path: Path,
    source_stem: str,
    box: tuple[float, float, float, float],
    rows: list[dict[str, str]],
) -> None:
    image = cv2.imread(str(original_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Unreadable image: {original_path}")

    # Horizontal flip changes only x-center and preserves a vertical door.
    flipped = cv2.flip(image, 1)
    flipped_name = f"door_train_{source_stem}_flip.png"
    cv2.imwrite(str(OUTPUT_DATASET / "images" / "train" / flipped_name), flipped)
    flipped_box = (1.0 - box[0], box[1], box[2], box[3])
    write_label(
        OUTPUT_DATASET / "labels" / "train" / f"{Path(flipped_name).stem}.txt",
        flipped_box,
    )
    rows.append(
        {
            "file": flipped_name,
            "split": "train",
            "source_split": "train",
            "augmentation": "horizontal_flip",
            "source_image": f"{source_stem}.png",
        }
    )

    # A deterministic low-light / bright-camera variant. The door box is
    # unchanged because this transform changes only appearance.
    number = int(source_stem) if source_stem.isdigit() else sum(map(ord, source_stem))
    alpha = 0.78 if number % 2 == 0 else 1.16
    beta = -12 if number % 2 == 0 else 8
    photo = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    if number % 3 == 0:
        photo = cv2.GaussianBlur(photo, (3, 3), 0)
    photo_name = f"door_train_{source_stem}_photo.png"
    cv2.imwrite(str(OUTPUT_DATASET / "images" / "train" / photo_name), photo)
    write_label(
        OUTPUT_DATASET / "labels" / "train" / f"{Path(photo_name).stem}.txt",
        box,
    )
    rows.append(
        {
            "file": photo_name,
            "split": "train",
            "source_split": "train",
            "augmentation": "photometric",
            "source_image": f"{source_stem}.png",
        }
    )


def draw_preview(
    samples: list[tuple[Path, tuple[float, float, float, float]]],
    output: Path,
) -> None:
    thumb_w, thumb_h, columns = 240, 340, 5
    rows = (len(samples) + columns - 1) // columns
    canvas = Image.new("RGB", (thumb_w * columns, thumb_h * rows), "white")
    draw = ImageDraw.Draw(canvas)

    for index, (path, box) in enumerate(samples):
        with Image.open(path) as image:
            image = image.convert("RGB")
            original_w, original_h = image.size
            image.thumbnail((thumb_w - 12, thumb_h - 42))
            cell_x = (index % columns) * thumb_w
            cell_y = (index // columns) * thumb_h
            x = cell_x + (thumb_w - image.width) // 2
            y = cell_y + 4
            scale_x, scale_y = image.width / original_w, image.height / original_h

            cx, cy, bw, bh = box
            x1 = (cx - bw / 2) * original_w * scale_x + x
            y1 = (cy - bh / 2) * original_h * scale_y + y
            x2 = (cx + bw / 2) * original_w * scale_x + x
            y2 = (cy + bh / 2) * original_h * scale_y + y

            canvas.paste(image, (x, y))
            draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
            draw.text((cell_x + 5, cell_y + thumb_h - 30), path.name, fill="black")

    canvas.save(output, quality=92)


def validate_dataset(door_rows: list[dict[str, str]]) -> dict[str, object]:
    summary: dict[str, object] = {"door_added": {}, "class_boxes": {}}
    for split in ("train", "val", "test"):
        image_dir = OUTPUT_DATASET / "images" / split
        label_dir = OUTPUT_DATASET / "labels" / split
        images = {
            path.stem
            for path in image_dir.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        }
        labels = {path.stem for path in label_dir.glob("*.txt")}
        if images != labels:
            raise RuntimeError(
                f"{split}: image/label mismatch, "
                f"images_only={len(images-labels)}, labels_only={len(labels-images)}"
            )

        class_counts: dict[int, int] = {}
        for label in label_dir.glob("*.txt"):
            for line in label.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                parts = line.split()
                # camera_v2 legitimately contains both detection boxes
                # (class + 4 values) and segmentation polygons
                # (class + x/y point pairs). Preserve and validate both.
                is_box = len(parts) == 5
                is_polygon = len(parts) >= 7 and len(parts) % 2 == 1
                if not (is_box or is_polygon):
                    raise RuntimeError(f"Invalid YOLO row: {label}: {line}")
                class_id = int(parts[0])
                values = [float(value) for value in parts[1:]]
                if class_id not in range(5) or any(not 0.0 <= value <= 1.0 for value in values):
                    raise RuntimeError(f"Invalid class/coordinates: {label}: {line}")
                class_counts[class_id] = class_counts.get(class_id, 0) + 1

        summary["door_added"][split] = sum(
            row["split"] == split for row in door_rows
        )
        summary["class_boxes"][split] = class_counts
        summary[f"{split}_images"] = len(images)
        summary[f"{split}_labels"] = len(labels)
    return summary


def main() -> None:
    if not BASE_DATASET.exists():
        raise SystemExit(f"Missing base dataset: {BASE_DATASET}")
    if OUTPUT_DATASET.exists():
        manifest_path = WORK_DIR / "manifest.csv"
        if not manifest_path.exists():
            raise SystemExit(
                f"Output already exists without a build manifest: {OUTPUT_DATASET}. "
                "It is intentionally not overwritten."
            )
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            existing_rows = list(csv.DictReader(handle))
        if len(existing_rows) != 560:
            raise SystemExit(
                f"Output build is incomplete: expected 560 door rows, "
                f"found {len(existing_rows)}"
            )
        summary = validate_dataset(existing_rows)
        summary["source_train_originals"] = 200
        summary["source_test_originals"] = 40
        summary["class_names"] = ["student", "phone", "paper", "note", "door"]
        (WORK_DIR / "audit.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (OUTPUT_DATASET / "README_DOOR.md").write_text(
            "\n".join(
                [
                    "# camera_v5_door",
                    "",
                    "This is a copy of camera_v2 with class 4 = door.",
                    "camera_v2 and saved baseline weights were not modified.",
                    "Door source: DoorDetect-Class-Dataset (cite the repository/paper).",
                    "Only door training images were augmented (flip + photometric).",
                    "Door validation and test images remain unaugmented.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"Finalized existing dataset: {OUTPUT_DATASET}")
        return

    for key, folder_id in FOLDER_IDS.items():
        destination = SOURCE_DIR / key
        # Always ask gdown to enumerate the full folder. With resume=True it
        # keeps complete files, while also repairing an interrupted download.
        print(f"Downloading/verifying {key}...")
        download_folder(folder_id, destination)

    train_images = source_files(SOURCE_DIR / "train_images")
    train_masks = source_files(SOURCE_DIR / "train_masks")
    test_images = source_files(SOURCE_DIR / "test_images")
    test_masks = source_files(SOURCE_DIR / "test_masks")

    train_common = train_images.keys() & train_masks.keys()
    test_common = test_images.keys() & test_masks.keys()
    if len(train_common) != 200 or len(test_common) != 40:
        raise RuntimeError(
            "Unexpected image/mask pairing: "
            f"train images={len(train_images)}, masks={len(train_masks)}, "
            f"paired={len(train_common)}; "
            f"test images={len(test_images)}, masks={len(test_masks)}, "
            f"paired={len(test_common)}"
        )
    # Ignore only unpaired scratch files. The 200/40 required source pairs
    # must still be present, so a genuinely missing download cannot slip by.
    train_images = {stem: train_images[stem] for stem in train_common}
    train_masks = {stem: train_masks[stem] for stem in train_common}
    test_images = {stem: test_images[stem] for stem in test_common}
    test_masks = {stem: test_masks[stem] for stem in test_common}

    # Copy the frozen four-class dataset into a new location.
    shutil.copytree(BASE_DATASET, OUTPUT_DATASET)
    for split in ("train", "val", "test"):
        (OUTPUT_DATASET / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DATASET / "labels" / split).mkdir(parents=True, exist_ok=True)

    shuffled_stems = sorted(train_images)
    random.Random(RANDOM_SEED).shuffle(shuffled_stems)
    validation_count = max(20, round(len(shuffled_stems) * 0.2))
    validation_stems = set(shuffled_stems[:validation_count])

    manifest_rows: list[dict[str, str]] = []
    preview_samples: list[tuple[Path, tuple[float, float, float, float]]] = []

    for stem in sorted(train_images):
        split = "val" if stem in validation_stems else "train"
        destination, box = add_original(
            train_images[stem],
            train_masks[stem],
            split,
            "train",
            manifest_rows,
        )
        if len(preview_samples) < 15:
            preview_samples.append((destination, box))
        if split == "train":
            add_training_augmentations(destination, stem, box, manifest_rows)

    for stem in sorted(test_images):
        destination, box = add_original(
            test_images[stem],
            test_masks[stem],
            "test",
            "test",
            manifest_rows,
        )
        if len(preview_samples) < 20:
            preview_samples.append((destination, box))

    yaml_path = OUTPUT_DATASET / "dataset.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {OUTPUT_DATASET.as_posix()}",
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

    with (WORK_DIR / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_rows[0].keys())
        writer.writeheader()
        writer.writerows(manifest_rows)

    draw_preview(preview_samples, WORK_DIR / "door_boxes_preview.jpg")
    summary = validate_dataset(manifest_rows)
    summary["source_train_originals"] = len(train_images)
    summary["source_test_originals"] = len(test_images)
    summary["class_names"] = ["student", "phone", "paper", "note", "door"]
    (WORK_DIR / "audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUTPUT_DATASET / "README_DOOR.md").write_text(
        "\n".join(
            [
                "# camera_v5_door",
                "",
                "This is a copy of camera_v2 with class 4 = door.",
                "camera_v2 and saved baseline weights were not modified.",
                "Door source: DoorDetect-Class-Dataset (cite the repository/paper).",
                "Only door training images were augmented (flip + photometric).",
                "Door validation and test images remain unaugmented.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Dataset: {OUTPUT_DATASET}")
    print(f"Preview: {WORK_DIR / 'door_boxes_preview.jpg'}")


if __name__ == "__main__":
    main()
