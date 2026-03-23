"""Verify custom_tomato_data_yolo against custom_tomato_data_coco source."""

import json
from collections import Counter
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
base = project_root / "data" / "custom_tomato_dataset"
coco_dir = base / "custom_tomato_data_coco"
yolo_dir = base / "custom_tomato_data_yolo"

with open(coco_dir / "_annotations.coco.json", encoding="utf-8") as f:
    coco = json.load(f)

ok = True

# 1. 파일 수 비교
coco_imgs = {img["file_name"] for img in coco["images"]}
yolo_imgs = {p.name for p in (yolo_dir / "images" / "train").iterdir() if p.suffix in (".jpg", ".png")}
yolo_lbls = {p.stem for p in (yolo_dir / "labels" / "train").iterdir() if p.suffix == ".txt"}

print("=== 1. 파일 수 ===")
print(f"  COCO images       : {len(coco_imgs)}")
print(f"  YOLO images/train : {len(yolo_imgs)}")
print(f"  YOLO labels/train : {len(yolo_lbls)}")

if coco_imgs != yolo_imgs:
    missing = coco_imgs - yolo_imgs
    extra = yolo_imgs - coco_imgs
    if missing:
        print(f"  [FAIL] YOLO에 없는 이미지: {len(missing)}개 (예: {list(missing)[:3]})")
    if extra:
        print(f"  [FAIL] COCO에 없는 이미지: {len(extra)}개 (예: {list(extra)[:3]})")
    ok = False
else:
    print("  이미지 파일 완전 일치")

img_stems = {Path(fn).stem for fn in coco_imgs}
if img_stems != yolo_lbls:
    missing_lbl = img_stems - yolo_lbls
    extra_lbl = yolo_lbls - img_stems
    if missing_lbl:
        print(f"  [FAIL] 라벨 누락: {len(missing_lbl)}개")
    if extra_lbl:
        print(f"  [FAIL] 여분 라벨: {len(extra_lbl)}개")
    ok = False
else:
    print("  이미지-라벨 1:1 매칭 확인")

# 2. 어노테이션 수 비교
print("\n=== 2. 어노테이션 수 ===")
coco_cat_counts = Counter(a["category_id"] for a in coco["annotations"])
print(f"  COCO: ripe(id=0)={coco_cat_counts[0]}, unripe(id=1)={coco_cat_counts[1]}, total={len(coco['annotations'])}")

yolo_class_counts = Counter()
yolo_total = 0
for lbl_path in (yolo_dir / "labels" / "train").glob("*.txt"):
    text = lbl_path.read_text(encoding="utf-8").strip()
    if not text:
        continue
    for line in text.split("\n"):
        cls = int(line.split()[0])
        yolo_class_counts[cls] += 1
        yolo_total += 1

print(f"  YOLO: ripe(cls=0)={yolo_class_counts[0]}, unripe(cls=1)={yolo_class_counts[1]}, total={yolo_total}")

if len(coco["annotations"]) != yolo_total:
    print(f"  [FAIL] 총 어노테이션 수 불일치! COCO={len(coco['annotations'])} vs YOLO={yolo_total}")
    ok = False
if coco_cat_counts[0] != yolo_class_counts[0]:
    print(f"  [FAIL] ripe 수 불일치!")
    ok = False
if coco_cat_counts[1] != yolo_class_counts[1]:
    print(f"  [FAIL] unripe 수 불일치!")
    ok = False
if len(coco["annotations"]) == yolo_total:
    print("  총 어노테이션 수 일치")

# 3. 이미지별 bbox 수 대조 (샘플)
print("\n=== 3. 이미지별 bbox 수 대조 ===")
anns_by_image = {}
img_map = {img["id"]: img for img in coco["images"]}
for ann in coco["annotations"]:
    fn = img_map[ann["image_id"]]["file_name"]
    anns_by_image.setdefault(fn, []).append(ann)

mismatch = 0
for fn, anns in anns_by_image.items():
    stem = Path(fn).stem
    lbl_path = yolo_dir / "labels" / "train" / (stem + ".txt")
    if not lbl_path.exists():
        mismatch += 1
        continue
    text = lbl_path.read_text(encoding="utf-8").strip()
    yolo_count = len(text.split("\n")) if text else 0
    if len(anns) != yolo_count:
        mismatch += 1

print(f"  bbox 수 불일치 이미지: {mismatch} / {len(anns_by_image)}")
if mismatch > 0:
    print(f"  [FAIL] {mismatch}개 이미지에서 bbox 수 불일치!")
    ok = False

# 4. YOLO label 값 범위 검증
print("\n=== 4. YOLO label 값 범위 검증 ===")
bad_values = 0
bad_classes = 0
for lbl_path in (yolo_dir / "labels" / "train").glob("*.txt"):
    text = lbl_path.read_text(encoding="utf-8").strip()
    if not text:
        continue
    for line in text.split("\n"):
        parts = line.split()
        cls = int(parts[0])
        if cls not in (0, 1):
            bad_classes += 1
        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
            bad_values += 1

print(f"  잘못된 class: {bad_classes}")
print(f"  범위 밖 좌표: {bad_values}")
if bad_classes > 0 or bad_values > 0:
    ok = False

# 5. dataset.yaml 확인
print("\n=== 5. dataset.yaml ===")
yaml_path = yolo_dir / "dataset.yaml"
print(f"  {yaml_path.read_text(encoding='utf-8')}")

print()
if ok:
    print("===== ALL CHECKS PASSED =====")
else:
    print("===== SOME CHECKS FAILED =====")
