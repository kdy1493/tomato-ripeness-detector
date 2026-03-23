"""Verify merged COCO JSON against source datasets."""

import json
from collections import Counter
from pathlib import Path

root = Path(__file__).resolve().parents[1] / "data" / "custom_tomato_dataset"

with open(root / "merged/_annotations.coco.json", encoding="utf-8") as f:
    merged = json.load(f)

print("=== 1. 병합 결과 기본 통계 ===")
print(f"  Categories : {merged['categories']}")
print(f"  Images     : {len(merged['images'])}")
print(f"  Annotations: {len(merged['annotations'])}")

cat_counts = Counter(a["category_id"] for a in merged["annotations"])
print(f"  category_id 분포: {dict(sorted(cat_counts.items()))}")

print("\n=== 2. 원본 합산 대조 ===")
sources = [
    "annotationcoco_ms/train/_annotations.coco.json",
    "anotationcoco_all/train/_annotations.coco.json",
    "tomatoannotation.coco/train/_annotations.coco.json",
]
total_imgs = 0
total_anns = 0
total_ripe = 0
total_unripe = 0
for s in sources:
    with open(root / s, encoding="utf-8") as f:
        c = json.load(f)
    n_imgs = len(c["images"])
    n_anns = len(c["annotations"])
    r = sum(1 for a in c["annotations"] if a["category_id"] == 1)
    u = sum(1 for a in c["annotations"] if a["category_id"] == 2)
    total_imgs += n_imgs
    total_anns += n_anns
    total_ripe += r
    total_unripe += u
    name = s.split("/")[0]
    print(f"  {name:30s}  imgs={n_imgs:4d}  anns={n_anns:5d}  ripe={r:4d}  unripe={u:5d}")

print(f"  {'합산':30s}  imgs={total_imgs:4d}  anns={total_anns:5d}  ripe={total_ripe:4d}  unripe={total_unripe:5d}")
print(f"  {'병합결과':28s}  imgs={len(merged['images']):4d}  anns={len(merged['annotations']):5d}  ripe={cat_counts.get(0,0):4d}  unripe={cat_counts.get(1,0):5d}")
print()
ok = True
if total_imgs != len(merged["images"]):
    print("  [FAIL] 이미지 수 불일치!")
    ok = False
if total_anns != len(merged["annotations"]):
    print("  [FAIL] 어노테이션 수 불일치!")
    ok = False
if total_ripe != cat_counts.get(0, 0):
    print("  [FAIL] ripe 어노테이션 수 불일치!")
    ok = False
if total_unripe != cat_counts.get(1, 0):
    print("  [FAIL] unripe 어노테이션 수 불일치!")
    ok = False

print("\n=== 3. ID 유일성 검증 ===")
img_ids = [img["id"] for img in merged["images"]]
ann_ids = [ann["id"] for ann in merged["annotations"]]
dup_imgs = len(img_ids) - len(set(img_ids))
dup_anns = len(ann_ids) - len(set(ann_ids))
print(f"  Image ID 중복: {dup_imgs}")
print(f"  Annotation ID 중복: {dup_anns}")
if dup_imgs > 0:
    print("  [FAIL] Image ID 중복 존재!")
    ok = False
if dup_anns > 0:
    print("  [FAIL] Annotation ID 중복 존재!")
    ok = False

print("\n=== 4. 파일명 유일성 & 실제 파일 존재 검증 ===")
filenames = [img["file_name"] for img in merged["images"]]
dup_files = len(filenames) - len(set(filenames))
print(f"  파일명 중복: {dup_files}")
if dup_files > 0:
    print("  [FAIL] 파일명 중복 존재!")
    ok = False

images_dir = root / "merged" / "images"
missing = [fn for fn in filenames if not (images_dir / fn).exists()]
print(f"  이미지 파일 누락: {len(missing)}")
if missing:
    print(f"  [FAIL] 누락 파일 예시: {missing[:5]}")
    ok = False

actual_files = list(images_dir.glob("*.*"))
extra = len(actual_files) - len(filenames)
print(f"  실제 파일 수: {len(actual_files)}, JSON 이미지 수: {len(filenames)}, 차이: {extra}")

print("\n=== 5. 어노테이션 참조 무결성 ===")
img_id_set = set(img_ids)
orphan_anns = [a for a in merged["annotations"] if a["image_id"] not in img_id_set]
print(f"  존재하지 않는 image_id를 참조하는 어노테이션: {len(orphan_anns)}")
if orphan_anns:
    print("  [FAIL] 고아 어노테이션 존재!")
    ok = False

invalid_cats = [a for a in merged["annotations"] if a["category_id"] not in (0, 1)]
print(f"  유효하지 않은 category_id 어노테이션: {len(invalid_cats)}")
if invalid_cats:
    print("  [FAIL] 잘못된 카테고리 존재!")
    ok = False

print("\n=== 6. bbox 유효성 검증 ===")
img_map = {img["id"]: img for img in merged["images"]}
bad_bbox = 0
for ann in merged["annotations"]:
    x, y, w, h = [float(v) for v in ann["bbox"]]
    img = img_map[ann["image_id"]]
    if w <= 0 or h <= 0:
        bad_bbox += 1
    elif x < 0 or y < 0 or x + w > img["width"] + 1 or y + h > img["height"] + 1:
        bad_bbox += 1
print(f"  비정상 bbox: {bad_bbox}")
if bad_bbox > 0:
    print(f"  [WARN] {bad_bbox}개 bbox가 범위를 벗어남")

print()
if ok:
    print("===== ALL CHECKS PASSED =====")
else:
    print("===== SOME CHECKS FAILED =====")
