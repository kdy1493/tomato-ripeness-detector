"""
Merge three custom tomato COCO datasets into a single COCO JSON.

Input datasets (all under data/custom_tomato_dataset/):
  annotationcoco_ms/train/      307 images
  anotationcoco_all/train/      797 images
  tomatoannotation.coco/train/  320 images

Each dataset has categories:
  id=0  tomatoes / tomato   <- NOT used in annotations, ignored
  id=1  ripe
  id=2  unripe

Output (data/custom_tomato_dataset/merged/):
  images/                   all 1,424 images copied here
  _annotations.coco.json    merged COCO annotation with 2 categories:
                              id=0  ripe
                              id=1  unripe
"""

import json
import shutil
from pathlib import Path


SOURCES = [
    "annotationcoco_ms/train",
    "anotationcoco_all/train",
    "tomatoannotation.coco/train",
]

# Original category_id -> new category_id
# category_id 0 (tomatoes/tomato) is unused; skip it.
# category_id 1 (ripe)   -> new id 0
# category_id 2 (unripe) -> new id 1
CAT_ID_REMAP = {1: 0, 2: 1}

NEW_CATEGORIES = [
    {"id": 0, "name": "ripe",   "supercategory": "tomato"},
    {"id": 1, "name": "unripe", "supercategory": "tomato"},
]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def merge(project_root: Path) -> None:
    dataset_root = project_root / "data" / "custom_tomato_dataset"
    out_dir = dataset_root / "merged"
    out_images_dir = out_dir / "images"
    out_images_dir.mkdir(parents=True, exist_ok=True)

    merged_images: list[dict] = []
    merged_annotations: list[dict] = []

    image_id_offset = 0
    ann_id_offset = 0

    for src_rel in SOURCES:
        src_dir = dataset_root / src_rel
        json_path = src_dir / "_annotations.coco.json"

        print(f"\nLoading {json_path.relative_to(project_root)}")
        coco = load_json(json_path)

        images = coco["images"]
        annotations = coco["annotations"]

        # Build old_image_id -> new_image_id mapping
        id_map: dict[int, int] = {}
        for img in images:
            new_id = img["id"] + image_id_offset
            id_map[img["id"]] = new_id
            merged_images.append({
                **img,
                "id": new_id,
            })

            # Copy image file
            src_img = src_dir / img["file_name"]
            dst_img = out_images_dir / img["file_name"]
            if not dst_img.exists():
                shutil.copy2(src_img, dst_img)

        # Remap annotations
        skipped = 0
        for ann in annotations:
            old_cat = ann["category_id"]
            if old_cat not in CAT_ID_REMAP:
                skipped += 1
                continue

            new_ann_id = ann["id"] + ann_id_offset
            new_img_id = id_map[ann["image_id"]]
            new_cat_id = CAT_ID_REMAP[old_cat]

            merged_annotations.append({
                **ann,
                "id": new_ann_id,
                "image_id": new_img_id,
                "category_id": new_cat_id,
            })

        max_img_id = max(img["id"] for img in images)
        max_ann_id = max(ann["id"] for ann in annotations) if annotations else 0

        print(f"  Images   : {len(images):4d}  (id offset +{image_id_offset})")
        print(f"  Anns     : {len(annotations):5d}  (id offset +{ann_id_offset}, skipped={skipped})")

        image_id_offset += max_img_id + 1
        ann_id_offset   += max_ann_id + 1

    merged_coco = {
        "info": {
            "description": "Merged custom tomato dataset (ripe / unripe)",
            "version": "1.0",
        },
        "licenses": [],
        "categories": NEW_CATEGORIES,
        "images": merged_images,
        "annotations": merged_annotations,
    }

    out_json = out_dir / "_annotations.coco.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(merged_coco, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}")
    print(f"Merged COCO saved : {out_json.relative_to(project_root)}")
    print(f"Total images      : {len(merged_images)}")
    print(f"Total annotations : {len(merged_annotations)}")
    print(f"Categories        : {[c['name'] for c in NEW_CATEGORIES]}")
    print(f"Images copied to  : {out_images_dir.relative_to(project_root)}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    merge(project_root)
