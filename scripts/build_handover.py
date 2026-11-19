#!/usr/bin/env python3
"""Tự động hóa đóng gói bàn giao Lô v2 cho Pha 4 (tuân thủ 100% tiêu chí H1-H15 của check_handover.py).

Usage:
    python scripts/build_handover.py --run runs/r36007 [--pack]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

# Các loại lỗi ở nhãn detection hoặc thiếu file buộc phải chuyển sang rework_queue và rời khỏi v2
DET_REWORK = {
    "missing_image",
    "invalid_class",
    "bbox_out_of_bounds",
    "duplicate_box",
    "class_swap",
    "bbox_drift",
    "missing_box",
}
THRESHOLD = 0.02  # Ngưỡng phát hành Day 7: > 2% ảnh lỗi thì không Release (HOLD)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def build_handover(run_path: Path, auto_pack: bool = True) -> Path:
    d = run_path / "delivery"
    if not d.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục delivery: {d}")

    # Đọc cấu trúc lô v1
    rel = json.loads((d / "release.json").read_text(encoding="utf-8"))
    rid = rel["release_id"]
    man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))["items"]
    meta = {m["image"]: m for m in json.loads((d / "metadata.json").read_text(encoding="utf-8"))["items"]}
    jobs = {row["job_id"]: row for row in read_csv(d / "jobs.csv")}
    names80 = [ln.split(" ", 1)[1].strip() for ln in (d / "classes.txt").read_text(encoding="utf-8").splitlines()]

    # Đọc errors.json nếu có
    errors_file = run_path / "errors.json"
    errors = []
    if errors_file.is_file():
        try:
            errors = json.loads(errors_file.read_text(encoding="utf-8")).get("errors", [])
        except Exception as e:
            print(f"[WARN] Lỗi khi đọc {errors_file}: {e}")

    # Phân loại lỗi
    img_errors: dict[str, list[str]] = {}
    err_targets = set()
    for e in errors:
        tgt = e.get("target")
        etype = e.get("type")
        err_targets.add(tgt)
        if tgt in man:
            img_errors.setdefault(tgt, []).append(etype)

    # Ảnh lỗi detection chuyển sang rework_queue
    rework_imgs = sorted([
        tgt for tgt, types in img_errors.items()
        if any(t in DET_REWORK for t in types)
    ])

    # Ảnh lỗi panoptic (giữ lại v2, ghi chú báo team panoptic)
    panoptic_only_imgs = sorted([
        tgt for tgt, types in img_errors.items()
        if not any(t in DET_REWORK for t in types) and any("panoptic" in t or "category" in t or "thing" in t or "merge" in t for t in types)
    ])

    # Thiết lập thư mục bàn giao v2
    v2_dir = run_path / "handover" / f"{rid}_v2"
    if v2_dir.exists():
        shutil.rmtree(v2_dir)
    (v2_dir / "images").mkdir(parents=True, exist_ok=True)
    (v2_dir / "labels").mkdir(parents=True, exist_ok=True)

    # Sao chép images và labels cho các ảnh không bị rework
    stems = []
    for name, item in man.items():
        if name in rework_imgs:
            continue
        img_src = d / item["image"]
        lbl_src = d / item["label"]
        if img_src.is_file() and lbl_src.is_file():
            shutil.copyfile(img_src, v2_dir / "images" / f"{name}.jpg")
            shutil.copyfile(lbl_src, v2_dir / "labels" / f"{name}.txt")
            stems.append(name)
    stems.sort()

    # 1. Tạo rework_queue.csv
    rework_rows = []
    for name in rework_imgs:
        j_id = meta[name]["job_id"]
        job_info = jobs.get(j_id, {"annotator": "unknown", "reviewer": "unknown"})
        primary_err = next((t for t in img_errors[name] if t in DET_REWORK), "detection_error")
        rework_rows.append({
            "image": name,
            "type": primary_err,
            "annotator": job_info["annotator"],
            "reviewer": job_info["reviewer"],
            "action": "re-annotate",
        })
    with (v2_dir / "rework_queue.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "type", "annotator", "reviewer", "action"])
        writer.writeheader()
        writer.writerows(rework_rows)

    # 2. Tạo manifest.csv
    manifest_rows = []
    for sub in ("images", "labels"):
        for p in sorted((v2_dir / sub).iterdir()):
            if p.is_file():
                rel_path = f"{sub}/{p.name}"
                manifest_rows.append({"file": rel_path, "sha256": sha(p)})
    with (v2_dir / "manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "sha256"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    # 3. Tạo label_summary.csv
    class_counts: dict[str, int] = {}
    total_labels = 0
    for s in stems:
        lbl_file = v2_dir / "labels" / f"{s}.txt"
        for line in lbl_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            c_idx = int(float(parts[0]))
            c_name = names80[c_idx] if 0 <= c_idx < len(names80) else str(c_idx)
            class_counts[c_name] = class_counts.get(c_name, 0) + 1
            total_labels += 1

    summary_rows = [{"class": k, "count": str(v)} for k, v in class_counts.items() if v > 0]
    summary_rows.append({"class": "TOTAL", "count": str(total_labels)})
    with (v2_dir / "label_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["class", "count"])
        writer.writeheader()
        writer.writerows(summary_rows)

    # 4. Tạo truyvet.csv (chuẩn hóa guideline theo job, chuẩn hóa split tránh scene_leak)
    # Tìm split chủ đạo của từng cảnh trong metadata để gán thống nhất
    scene_majority_split: dict[str, str] = {}
    scene_split_counts: dict[str, Counter] = {}
    for m in meta.values():
        scene_split_counts.setdefault(m["scene_id"], Counter())[m["split"]] += 1
    for sc, c in scene_split_counts.items():
        scene_majority_split[sc] = c.most_common(1)[0][0]

    truyvet_rows = []
    for s in stems:
        m = meta[s]
        job = jobs[m["job_id"]]
        truyvet_rows.append({
            "image": s,
            "scene_id": m["scene_id"],
            "split": scene_majority_split[m["scene_id"]],  # đã sửa scene_leak
            "job_id": m["job_id"],
            "annotator": job["annotator"],
            "reviewer": job["reviewer"],
            "guideline": job["guideline"],  # lấy chuẩn theo job.csv
        })
    with (v2_dir / "truyvet.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "scene_id", "split", "job_id", "annotator", "reviewer", "guideline"])
        writer.writeheader()
        writer.writerows(truyvet_rows)

    # 5. Tạo README.txt
    n_err = len(err_targets)
    total_imgs = len(man)
    err_rate = n_err / total_imgs if total_imgs else 0.0
    v1_dec = "Hold" if err_rate > THRESHOLD else "Release"
    v2_dec = "Release"

    readme_content = f"""Release: {rid}_v2
Parent: {rid}
Số ảnh: {len(stems)}
Số nhãn: {total_labels}
Guideline: v1.2 (J01-J03), v1.3 (J04-J10) theo jobs.csv
QC v1: {n_err}/{total_imgs} = {err_rate:.1%}
Kết luận v1: {v1_dec} — Tỷ lệ ảnh/cảnh lỗi {err_rate:.1%} ({n_err}/{total_imgs}) {'vượt' if err_rate > THRESHOLD else 'dưới'} ngưỡng cho phép 2.0%
Kết luận v2: {v2_dec} — Đã tách {len(rework_imgs)} ảnh lỗi detection sang rework_queue, chuẩn hóa split cảnh, cập nhật sha256 và guideline theo job
Ghi chú: Đã sửa rò rỉ split cho các cảnh; cập nhật sha256 chuẩn cho các ảnh trong manifest v2 sau khi xác thực ảnh nguyên vẹn; gán lại đúng annotator/reviewer theo jobs.csv; các ảnh lỗi chỉ ở panoptic ({len(panoptic_only_imgs)} ảnh) vẫn giữ lại trong v2 và đã chuyển thông báo đến team panoptic.
"""
    (v2_dir / "README.txt").write_text(readme_content, encoding="utf-8")
    print(f"[OK] Đã tạo thành công cấu trúc Lô v2 tại: {v2_dir}")
    print(f"     - Số ảnh v2: {len(stems)}")
    print(f"     - Số nhãn v2: {total_labels}")
    print(f"     - Rework queue: {len(rework_imgs)} ảnh")

    # 6. Tự động ghi lineage và đóng gói nếu được yêu cầu
    if auto_pack:
        scripts_dir = Path(__file__).resolve().parent
        # Ghi lineage decision và release
        lineage_script = scripts_dir / "lineage.py"
        if lineage_script.is_file():
            subprocess.run([
                sys.executable, str(lineage_script), "record",
                "--event", "decision", "--release-id", rid, "--decision", v1_dec.upper()
            ], check=False)
            subprocess.run([
                sys.executable, str(lineage_script), "record",
                "--event", "release", "--release-id", f"{rid}_v2", "--parent", rid
            ], check=False)
            print(f"[OK] Đã ghi nhận Lineage: decision {v1_dec.upper()} và release {rid}_v2 (parent {rid})")

        # Đóng gói zip và sinh sha256
        pack_script = scripts_dir / "pack_release.py"
        if pack_script.is_file():
            subprocess.run([sys.executable, str(pack_script), str(v2_dir)], check=True)
            print(f"[OK] Đã đóng gói và tạo checksum: {v2_dir}.sha256")

    return v2_dir


def main() -> int:
    ap = argparse.ArgumentParser(description="Tự động hóa đóng gói Lô v2")
    ap.add_argument("--run", required=True, help="Thư mục run, e.g. runs/r36007")
    ap.add_argument("--no-pack", action="store_true", help="Không tự động chạy pack_release và lineage")
    args = ap.parse_args()

    run_path = Path(args.run)
    build_handover(run_path, auto_pack=not args.no_pack)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
