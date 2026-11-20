#!/usr/bin/env python3
"""Công cụ trực quan hóa (visualize) các ảnh ứng viên để nhóm review bằng mắt.

Vẽ box Detection (Màu đỏ/xanh dương) và Box Panoptic (Màu xanh lá) lên ảnh,
lưu vào thư mục review_images/ để mở xem trực tiếp.

Usage:
    python scripts/visualize_candidates.py --run runs/r36007
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser(description="Vẽ nhãn lên ảnh ứng viên")
    parser.add_argument("--run", default="runs/r36007", help="Đường dẫn run")
    args = parser.parse_args()

    run_dir = Path(args.run)
    cand_path = run_dir / "candidates.json"
    if not cand_path.is_file():
        print(f"Không tìm thấy file {cand_path}")
        return

    data = json.loads(cand_path.read_text(encoding="utf-8"))
    candidates = data.get("candidates", [])
    print(f"Tìm thấy {len(candidates)} ứng viên trong {cand_path}")

    out_dir = run_dir / "review_images"
    out_dir.mkdir(parents=True, exist_ok=True)

    img_dir = run_dir / "delivery" / "images"

    for idx, cand in enumerate(candidates, 1):
        img_name = cand["image"]
        img_path = img_dir / f"{img_name}.jpg"
        if not img_path.is_file():
            continue

        im = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(im)
        w, h = im.size

        cand_type = cand.get("type", "")

        # Vẽ box Detection (Đỏ)
        if "det_bbox" in cand:
            b = cand["det_bbox"]
            det_c = cand.get("det_class", "det")
            # bbox: xmin, ymin, xmax, ymax
            draw.rectangle([b[0], b[1], b[2], b[3]], outline="red", width=3)
            draw.text((b[0] + 3, b[1] + 3), f"DET: {det_c}", fill="red")

        # Vẽ box Panoptic (Xanh lá)
        if "pan_bbox" in cand:
            pb = cand["pan_bbox"]
            pan_c = cand.get("pan_class", cand.get("class", "pan"))
            # pan_bbox có thể là [xmin, ymin, xmax, ymax] hoặc [x, y, w, h]
            # trong candidates của nhóm: [xmin, ymin, xmax, ymax]
            draw.rectangle([pb[0], pb[1], pb[2], pb[3]], outline="lime", width=2)
            draw.text((pb[0] + 3, pb[3] - 15), f"PAN: {pan_c}", fill="lime")

        # Tiêu đề ghi chú ở mép trên ảnh
        banner_text = f"[{idx}/{len(candidates)}] {img_name} | Type: {cand_type}"
        draw.rectangle([0, 0, w, 24], fill="black")
        draw.text((5, 5), banner_text, fill="yellow")

        out_file = out_dir / f"{idx:02d}_{img_name}_{cand_type}.jpg"
        im.save(out_file)
        print(f"  [{idx:02d}] Đã lưu: {out_file.name}")

    print(f"\n[OK] Toàn bộ ảnh đã được vẽ box và lưu tại: {out_dir}")


if __name__ == "__main__":
    main()
