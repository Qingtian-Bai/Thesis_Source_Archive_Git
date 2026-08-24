"""Build a timestamped contact sheet from the hybrid JSONL evidence log."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--alert-suffix", default="PHONE")
    args = parser.parse_args()

    evidence = Path(args.evidence_dir).resolve()
    log_path = next(evidence.glob("log_*.jsonl"))
    rows = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        if item.get("alert_type", "").endswith(args.alert_suffix):
            rows.append(item)
    rows.sort(key=lambda item: float(item["source_time_seconds"]))

    columns = 4
    tile_w, image_h, label_h = 384, 216, 42
    rows_count = (len(rows) + columns - 1) // columns
    canvas = np.full((rows_count * (image_h + label_h), columns * tile_w, 3), 245, np.uint8)
    for index, item in enumerate(rows):
        image = cv2.imread(str(item["image_path"]))
        if image is None:
            continue
        image = cv2.resize(image, (tile_w, image_h), interpolation=cv2.INTER_AREA)
        row, column = divmod(index, columns)
        x, y = column * tile_w, row * (image_h + label_h)
        canvas[y:y + image_h, x:x + tile_w] = image
        detection = (item.get("detections") or [{}])[0]
        label = (
            f"#{index + 1:02d} t={float(item['source_time_seconds']):.3f}s "
            f"tracks={detection.get('track_ids', [])}"
        )
        cv2.putText(canvas, label, (x + 6, y + image_h + 27), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, (20, 20, 20), 1, cv2.LINE_AA)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), canvas):
        raise RuntimeError(output)
    print(output)


if __name__ == "__main__":
    main()
