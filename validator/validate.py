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
import math
import sys
from pathlib import Path
from typing import Any

# Cho phép import scripts/pngutil.py từ thư mục gốc repo
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.pngutil import read_rgb, segment_rgb


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bbox_iou(b1: list[float], b2: list[float]) -> float:
    ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0.0 else 0.0


def union_bbox(b1: list[float], b2: list[float]) -> list[float]:
    return [
        min(b1[0], b2[0]),
        min(b1[1], b2[1]),
        max(b1[2], b2[2]),
        max(b1[3], b2[3]),
    ]


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


def find_candidates(
    delivery_dir: Path, rule_errors: list[dict[str, str]]
) -> list[dict[str, Any]]:
    """Đối chiếu nhãn detection (YOLO) và panoptic (JSON + mask PNG) để sinh ứng viên."""
    rule_imgs = {e["target"] for e in rule_errors}

    classes: dict[int, str] = {}
    classes_path = delivery_dir / "classes.txt"
    for line in classes_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            cid_str, name = line.strip().split(" ", 1)
            classes[int(cid_str)] = name.strip()

    panoptic_path = delivery_dir / "panoptic.json"
    panoptic = json.loads(panoptic_path.read_text(encoding="utf-8"))
    cats = {int(c["id"]): c for c in panoptic.get("categories", [])}
    imgs = {
        Path(i["file_name"]).stem: (int(i["width"]), int(i["height"]))
        for i in panoptic.get("images", [])
    }
    anns = {
        Path(a["file_name"]).stem: a.get("segments_info", [])
        for a in panoptic.get("annotations", [])
    }

    candidates: list[dict[str, Any]] = []

    for stem, (width, height) in sorted(imgs.items()):
        if stem in rule_imgs:
            continue
        lbl_path = delivery_dir / "labels" / f"{stem}.txt"
        msk_path = delivery_dir / "masks" / f"{stem}.png"
        if not lbl_path.exists() or not msk_path.exists():
            continue

        boxes: list[dict[str, Any]] = []
        for idx, line in enumerate(lbl_path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            cid = int(parts[0])
            cx = float(parts[1]) * width
            cy = float(parts[2]) * height
            bw = float(parts[3]) * width
            bh = float(parts[4]) * height
            boxes.append(
                {
                    "idx": idx,
                    "cls": classes.get(cid, "?"),
                    "bbox": [cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0],
                    "area": bw * bh,
                }
            )

        segs: list[dict[str, Any]] = []
        for s_idx, seg in enumerate(anns.get(stem, [])):
            cat = cats.get(int(seg["category_id"]))
            if not cat or not cat.get("isthing") or seg.get("iscrowd", 0):
                continue
            bx, by, bw, bh = seg["bbox"]
            segs.append(
                {
                    "s_idx": s_idx,
                    "id": int(seg["id"]),
                    "rgb": segment_rgb(int(seg["id"])),
                    "cls": cat["name"],
                    "bbox": [float(bx), float(by), float(bx + bw), float(by + bh)],
                    "area": int(seg["area"]),
                }
            )

        if not segs and not boxes:
            continue

        _, _, pixels = read_rgb(msk_path)
        stride = width * 3

        box_seg_pixels: list[collections.Counter[tuple[int, int, int]]] = []
        for b in boxes:
            ix1 = max(0, min(width, int(math.floor(b["bbox"][0]))))
            iy1 = max(0, min(height, int(math.floor(b["bbox"][1]))))
            ix2 = max(0, min(width, int(math.ceil(b["bbox"][2]))))
            iy2 = max(0, min(height, int(math.ceil(b["bbox"][3]))))
            sub = bytearray()
            row_len = (ix2 - ix1) * 3
            if row_len > 0 and iy2 > iy1:
                for y in range(iy1, iy2):
                    start = y * stride + ix1 * 3
                    sub += pixels[start : start + row_len]
            counts = collections.Counter(zip(sub[0::3], sub[1::3], sub[2::3]))
            box_seg_pixels.append(counts)

        # Bước 1: Ghép cặp 1-1 các (box, segment) cùng lớp khớp khít nhau
        clean_pairs: list[tuple[float, int, int]] = []
        for b_i, b in enumerate(boxes):
            for s_i, s in enumerate(segs):
                if b["cls"] != s["cls"]:
                    continue
                ov = bbox_iou(b["bbox"], s["bbox"])
                max_edge_diff = max(abs(b["bbox"][k] - s["bbox"][k]) for k in range(4))
                if ov >= 0.78 or max_edge_diff <= 2.5:
                    clean_pairs.append((ov, b_i, s_i))

        clean_pairs.sort(reverse=True)
        matched_b: set[int] = set()
        matched_s: set[int] = set()
        for _, b_i, s_i in clean_pairs:
            if b_i not in matched_b and s_i not in matched_s:
                matched_b.add(b_i)
                matched_s.add(s_i)

        unmatched_b = [b_i for b_i in range(len(boxes)) if b_i not in matched_b]
        unmatched_s = [
            s_i
            for s_i in range(len(segs))
            if s_i not in matched_s and segs[s_i]["area"] >= 350
        ]

        if not unmatched_s:
            continue

        used_s: set[int] = set()

        # A. Kiểm tra class_swap / category_swap: trùng vị trí nhưng khác tên lớp
        for b_i in unmatched_b:
            b = boxes[b_i]
            for s_i in unmatched_s:
                if s_i in used_s:
                    continue
                s = segs[s_i]
                if b["cls"] == s["cls"]:
                    continue
                ov = bbox_iou(b["bbox"], s["bbox"])
                frac = (
                    box_seg_pixels[b_i].get(s["rgb"], 0) / s["area"]
                    if s["area"] > 0
                    else 0.0
                )
                if ov >= 0.70 or (ov >= 0.55 and frac >= 0.80):
                    candidates.append(
                        {
                            "image": stem,
                            "type": "class_swap_or_category_swap",
                            "det_class": b["cls"],
                            "pan_class": s["cls"],
                            "det_bbox": [round(x, 1) for x in b["bbox"]],
                            "pan_bbox": [round(x, 1) for x in s["bbox"]],
                            "iou": round(ov, 3),
                            "mask_in_box": round(frac, 3),
                            "area": s["area"],
                        }
                    )
                    used_s.add(s_i)

        # B. Kiểm tra instance_merge: 1 segment panoptic phủ 2 box detection cùng lớp
        for s_i in unmatched_s:
            if s_i in used_s:
                continue
            s = segs[s_i]
            cov_b: list[tuple[int, dict[str, Any], float, float]] = []
            for b_i in unmatched_b:
                b = boxes[b_i]
                if b["cls"] != s["cls"]:
                    continue
                pix = box_seg_pixels[b_i].get(s["rgb"], 0)
                share = pix / s["area"] if s["area"] > 0 else 0.0
                fill = pix / b["area"] if b["area"] > 0 else 0.0
                if share >= 0.04 and fill >= 0.10:
                    cov_b.append((b_i, b, share, fill))
            if len(cov_b) >= 2:
                found_merge = False
                for i in range(len(cov_b)):
                    if found_merge:
                        break
                    for j in range(i + 1, len(cov_b)):
                        _, b1, sh1, _ = cov_b[i]
                        _, b2, sh2, _ = cov_b[j]
                        ub = union_bbox(b1["bbox"], b2["bbox"])
                        u_iou = bbox_iou(ub, s["bbox"])
                        if u_iou >= 0.75 and (sh1 + sh2) >= 0.70:
                            candidates.append(
                                {
                                    "image": stem,
                                    "type": "instance_merge",
                                    "class": s["cls"],
                                    "box1": [round(x, 1) for x in b1["bbox"]],
                                    "box2": [round(x, 1) for x in b2["bbox"]],
                                    "pan_bbox": [round(x, 1) for x in s["bbox"]],
                                    "union_iou": round(u_iou, 3),
                                    "area": s["area"],
                                }
                            )
                            used_s.add(s_i)
                            found_merge = True
                            break

        # C. Kiểm tra bbox_drift: box và segment cùng lớp nhưng lệch vị trí
        for s_i in unmatched_s:
            if s_i in used_s:
                continue
            s = segs[s_i]
            for b_i in unmatched_b:
                b = boxes[b_i]
                if b["cls"] != s["cls"]:
                    continue
                ov = bbox_iou(b["bbox"], s["bbox"])
                frac = (
                    box_seg_pixels[b_i].get(s["rgb"], 0) / s["area"]
                    if s["area"] > 0
                    else 0.0
                )
                if 0.10 <= ov < 0.78 and frac >= 0.15:
                    candidates.append(
                        {
                            "image": stem,
                            "type": "bbox_drift",
                            "class": s["cls"],
                            "det_bbox": [round(x, 1) for x in b["bbox"]],
                            "pan_bbox": [round(x, 1) for x in s["bbox"]],
                            "iou": round(ov, 3),
                            "mask_in_box": round(frac, 3),
                            "area": s["area"],
                        }
                    )
                    used_s.add(s_i)
                    break

        # D. Kiểm tra missing_box: segment thing lớn không có box tương ứng
        for s_i in unmatched_s:
            if s_i in used_s:
                continue
            s = segs[s_i]
            max_ov_unmatched = max(
                (bbox_iou(boxes[b_i]["bbox"], s["bbox"]) for b_i in unmatched_b),
                default=0.0,
            )
            if max_ov_unmatched < 0.15:
                candidates.append(
                    {
                        "image": stem,
                        "type": "missing_box",
                        "class": s["cls"],
                        "pan_bbox": [round(x, 1) for x in s["bbox"]],
                        "area": s["area"],
                    }
                )
                used_s.add(s_i)

    return candidates


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
        candidates = find_candidates(args.delivery, rule_errors)
        args.candidates.parent.mkdir(parents=True, exist_ok=True)
        args.candidates.write_text(
            json.dumps({"candidates": candidates}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
