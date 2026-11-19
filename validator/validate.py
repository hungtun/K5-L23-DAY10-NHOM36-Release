#!/usr/bin/env python3
"""Validator cho Day 10 Lab — Bắt lỗi luật (Pha 1) và sinh ứng viên (Pha 2).

Chỉ dùng thư viện chuẩn Python và scripts/pngutil.py.
Chỉ đọc thư mục delivery được truyền vào.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import sys
from pathlib import Path

# Cho phép import scripts/pngutil.py từ thư mục gốc repo
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.pngutil import read_rgb, segment_rgb


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_rules(delivery_dir: Path) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add_error(err_type: str, target: str) -> None:
        key = (err_type, target)
        if key not in seen:
            seen.add(key)
            errors.append({"type": err_type, "target": target})

    # 1. Đọc manifest.json
    manifest_path = delivery_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    num_classes = int(manifest.get("num_classes", 80))
    manifest_items: dict[str, dict[str, str]] = manifest.get("items", {})

    # 2. Đọc jobs.csv
    job_guidelines: dict[str, str] = {}
    jobs_path = delivery_dir / "jobs.csv"
    with jobs_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            job_guidelines[row["job_id"].strip()] = row["guideline"].strip()

    # 3. Đọc metadata.json
    metadata_path = delivery_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    scene_splits: dict[str, set[str]] = collections.defaultdict(set)

    for item in metadata.get("items", []):
        img_id = item["image"]
        scene_id = item["scene_id"]
        split = item["split"]
        job_id = item["job_id"]
        guideline_ver = item["guideline_version"]

        scene_splits[scene_id].add(split)
        expected_guideline = job_guidelines.get(job_id)
        if expected_guideline is not None and guideline_ver != expected_guideline:
            add_error("guideline_mismatch", img_id)

    for scene_id, splits in scene_splits.items():
        if len(splits) > 1:
            add_error("scene_leak", scene_id)

    # 4. Đọc panoptic.json
    panoptic_path = delivery_dir / "panoptic.json"
    panoptic = json.loads(panoptic_path.read_text(encoding="utf-8"))
    valid_cat_ids = {int(c["id"]) for c in panoptic.get("categories", [])}

    image_dims: dict[str, tuple[int, int]] = {}
    for img_info in panoptic.get("images", []):
        stem = Path(img_info["file_name"]).stem
        image_dims[stem] = (int(img_info["width"]), int(img_info["height"]))

    # 5. Kiểm tra File (missing_image, wrong_hash) và Nhãn Detection
    for img_id, info in manifest_items.items():
        img_path = delivery_dir / info["image"]
        lbl_path = delivery_dir / info["label"]
        msk_path = delivery_dir / info["mask"]

        if not img_path.exists():
            add_error("missing_image", img_id)
        else:
            if sha256_file(img_path) != info.get("image_sha256"):
                add_error("wrong_hash", img_id)

        if lbl_path.exists():
            if sha256_file(lbl_path) != info.get("label_sha256"):
                add_error("wrong_hash", img_id)
        if msk_path.exists():
            if sha256_file(msk_path) != info.get("mask_sha256"):
                add_error("wrong_hash", img_id)

        # Kiểm tra nhãn detection
        if lbl_path.exists():
            width, height = image_dims.get(img_id, (640, 480))
            boxes_px: list[tuple[int, float, float, float, float]] = []
            lines = lbl_path.read_text(encoding="utf-8").splitlines()
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                cx, cy, w, h = (
                    float(parts[1]),
                    float(parts[2]),
                    float(parts[3]),
                    float(parts[4]),
                )

                if cls_id < 0 or cls_id >= num_classes:
                    add_error("invalid_class", img_id)

                x1 = cx - w / 2.0
                y1 = cy - h / 2.0
                x2 = cx + w / 2.0
                y2 = cy + h / 2.0
                eps = 1e-5
                if (
                    w <= 0.0
                    or h <= 0.0
                    or x1 < -eps
                    or y1 < -eps
                    or x2 > 1.0 + eps
                    or y2 > 1.0 + eps
                ):
                    add_error("bbox_out_of_bounds", img_id)

                boxes_px.append((cls_id, cx * width, cy * height, w * width, h * height))

            # Kiểm tra duplicate_box: cùng lớp, cùng kích thước, tâm lệch ~1-2 pixel
            n_boxes = len(boxes_px)
            found_dup = False
            for i in range(n_boxes):
                if found_dup:
                    break
                c1, cx1, cy1, w1, h1 = boxes_px[i]
                for j in range(i + 1, n_boxes):
                    c2, cx2, cy2, w2, h2 = boxes_px[j]
                    if c1 != c2:
                        continue
                    if (
                        abs(w1 - w2) <= 1.5
                        and abs(h1 - h2) <= 1.5
                        and abs(cx1 - cx2) <= 3.5
                        and abs(cy1 - cy2) <= 3.5
                    ):
                        add_error("duplicate_box", img_id)
                        found_dup = True
                        break

    # 6. Kiểm tra Panoptic (invalid_category_id, thing_overlap)
    for ann in panoptic.get("annotations", []):
        img_id = Path(ann["file_name"]).stem
        segments = ann.get("segments_info", [])

        for seg in segments:
            if int(seg["category_id"]) not in valid_cat_ids:
                add_error("invalid_category_id", img_id)

        mask_rel = manifest_items.get(img_id, {}).get("mask", f"masks/{ann['file_name']}")
        mask_path = delivery_dir / mask_rel
        if mask_path.exists() and segments:
            _, _, pixels = read_rgb(mask_path)
            rgb_counts = collections.Counter(
                zip(pixels[0::3], pixels[1::3], pixels[2::3])
            )
            for seg in segments:
                seg_id = int(seg["id"])
                if seg_id == 0:
                    continue
                declared_area = int(seg["area"])
                actual_area = rgb_counts.get(segment_rgb(seg_id), 0)
                if actual_area != declared_area:
                    add_error("thing_overlap", img_id)
                    break

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate dataset delivery.")
    parser.add_argument("delivery", type=Path, help="Path to delivery directory")
    parser.add_argument("--out", type=Path, required=True, help="Output errors.json path")
    parser.add_argument(
        "--candidates",
        type=Path,
        default=None,
        help="Optional output candidates.json path (Phase 2)",
    )
    args = parser.parse_args()

    rule_errors = check_rules(args.delivery)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"errors": rule_errors}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    if args.candidates is not None:
        args.candidates.parent.mkdir(parents=True, exist_ok=True)
        args.candidates.write_text(
            json.dumps({"candidates": []}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()

