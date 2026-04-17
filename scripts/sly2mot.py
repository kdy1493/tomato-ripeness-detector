#!/usr/bin/env python3
"""
Supervisely 검수 완료 ann.json → MOT Challenge CSV 변환

Supervisely에서 Export → "Supervisely JSON" 으로 다운받은 ann.json을
benchmark.py --gt 에 넘길 수 있는 gt_mot.csv 로 변환합니다.

사용법:
    python scripts/sly2mot.py
    python scripts/sly2mot.py --sly-ann downloads/ann.json
    python scripts/sly2mot.py --sly-ann downloads/ann.json --out tracking_result/gt_mot.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

CONFIG = {
    "sly_ann": "tracking_result/supervisely_gt/ann.json",
    "out_csv": "tracking_result/gt_mot.csv",
}

CLASS_TITLE_TO_ID = {"ripe": 0, "unripe": 1}


def convert(sly_ann_path: Path, out_csv_path: Path) -> None:
    with open(sly_ann_path, encoding="utf-8") as f:
        ann = json.load(f)

    # object key → (stable_id, class_id)
    obj_map: dict[str, tuple[int, int]] = {}
    for idx, obj in enumerate(ann.get("objects", []), start=1):
        cid = CLASS_TITLE_TO_ID.get(obj["classTitle"], -1)
        obj_map[obj["key"]] = (idx, cid)

    rows = []
    for frame in ann.get("frames", []):
        frame_idx = frame["index"]    # 0-based
        frame_id  = frame_idx + 1     # MOT: 1-based

        for fig in frame.get("figures", []):
            obj_key = fig["objectKey"]
            if obj_key not in obj_map:
                continue
            sid, cid = obj_map[obj_key]
            if cid < 0:
                continue

            ext = fig["geometry"]["points"]["exterior"]
            x1, y1 = ext[0]
            x2, y2 = ext[1]
            rows.append((
                frame_id,
                sid,
                round(min(x1, x2), 2),
                round(min(y1, y2), 2),
                round(abs(x2 - x1), 2),
                round(abs(y2 - y1), 2),
                1,
                cid,
            ))

    rows.sort(key=lambda r: (r[0], r[7], r[1]))

    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame_id", "track_id", "x", "y", "w", "h", "conf", "class_id"])
        w.writerows(rows)

    n_r = len({sid for _, sid, *_, cid in rows if cid == 0})
    n_u = len({sid for _, sid, *_, cid in rows if cid == 1})
    print(f"변환 완료: {out_csv_path}")
    print(f"  총 행: {len(rows)}  |  ripe tracks={n_r}  unripe tracks={n_u}")
    print(f"\n벤치마크 실행:")
    print(f"  python scripts/benchmark.py --gt {out_csv_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Supervisely ann.json → MOT CSV")
    p.add_argument("--sly-ann", type=str, default=None, dest="sly_ann",
                   help="Supervisely ann.json 경로 (repo-relative)")
    p.add_argument("--out",     type=str, default=None,
                   help="출력 CSV 경로 (repo-relative)")
    args = p.parse_args()

    cfg = dict(CONFIG)
    if args.sly_ann: cfg["sly_ann"] = args.sly_ann
    if args.out:     cfg["out_csv"] = args.out

    convert(
        sly_ann_path = REPO_ROOT / cfg["sly_ann"],
        out_csv_path = REPO_ROOT / cfg["out_csv"],
    )


if __name__ == "__main__":
    main()
