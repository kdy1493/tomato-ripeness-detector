"""
Convert merged_dataset (COCO format) to YOLO format.

Input structure:
  data/merged_dataset/
    train/_annotations.coco.json + *.jpg
    valid/_annotations.coco.json + *.jpg
    test/_annotations.coco.json + *.jpg

Output structure:
  data/merged_dataset/
    dataset.yaml
    images/train/*.jpg  (symlink or copy)
    images/val/*.jpg
    images/test/*.jpg
    labels/train/*.txt
    labels/val/*.txt
    labels/test/*.txt
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


def load_coco_json(json_path: Path) -> Dict:
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def coco_bbox_to_yolo(
    bbox: List[float], img_w: int, img_h: int
) -> Tuple[float, float, float, float]:
    """Convert COCO [x, y, w, h] (absolute) to YOLO [cx, cy, w, h] (normalized)."""
    x, y, w, h = bbox
    cx = (x + w / 2.0) / img_w
    cy = (y + h / 2.0) / img_h
    nw = w / img_w
    nh = h / img_h
    return cx, cy, nw, nh


def convert_split(
    coco_data: Dict,
    src_image_dir: Path,
    dst_images_dir: Path,
    dst_labels_dir: Path,
    cat_id_to_yolo: Dict[int, int],
) -> int:
    """Convert one split (train/val/test) from COCO to YOLO format."""
    dst_images_dir.mkdir(parents=True, exist_ok=True)
    dst_labels_dir.mkdir(parents=True, exist_ok=True)

    images_by_id = {img["id"]: img for img in coco_data["images"]}
    anns_by_image: Dict[int, List] = {}
    for ann in coco_data.get("annotations", []):
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    converted = 0
    for img_id, img_info in images_by_id.items():
        file_name = img_info["file_name"]
        src_img = src_image_dir / file_name
        if not src_img.exists():
            continue

        dst_img = dst_images_dir / file_name
        shutil.copy2(src_img, dst_img)

        img_w = img_info["width"]
        img_h = img_info["height"]

        label_lines = []
        for ann in anns_by_image.get(img_id, []):
            cat_id = ann["category_id"]
            if cat_id not in cat_id_to_yolo:
                continue
            yolo_cls = cat_id_to_yolo[cat_id]
            cx, cy, nw, nh = coco_bbox_to_yolo(ann["bbox"], img_w, img_h)
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            nw = max(0.0, min(1.0, nw))
            nh = max(0.0, min(1.0, nh))
            if nw <= 0 or nh <= 0:
                continue
            label_lines.append(f"{yolo_cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        label_path = dst_labels_dir / (Path(file_name).stem + ".txt")
        label_path.write_text("\n".join(label_lines), encoding="utf-8")
        converted += 1

    return converted


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    dataset_root = project_root / "data" / "merged_dataset"

    splits_mapping = {
        "train": "train",
        "valid": "val",
        "test": "test",
    }

    first_coco = load_coco_json(dataset_root / "train" / "_annotations.coco.json")
    categories = first_coco["categories"]
    sorted_cats = sorted(categories, key=lambda c: c["id"])
    cat_id_to_yolo = {cat["id"]: idx for idx, cat in enumerate(sorted_cats)}
    class_names = [cat["name"] for cat in sorted_cats]

    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"Category ID mapping to YOLO: {cat_id_to_yolo}")

    for src_split, dst_split in splits_mapping.items():
        src_dir = dataset_root / src_split
        coco_json = src_dir / "_annotations.coco.json"
        if not coco_json.exists():
            print(f"Warning: {coco_json} not found, skipping {src_split}")
            continue

        coco_data = load_coco_json(coco_json)
        dst_images = dataset_root / "images" / dst_split
        dst_labels = dataset_root / "labels" / dst_split

        count = convert_split(coco_data, src_dir, dst_images, dst_labels, cat_id_to_yolo)
        print(f"Converted {src_split} -> {dst_split}: {count} images")

    yaml_content = f"""\
# Ultralytics YOLO dataset config for merged_dataset
path: {dataset_root.as_posix()}
train: images/train
val: images/val
test: images/test

nc: {len(class_names)}
names: {class_names}
"""
    yaml_path = dataset_root / "dataset.yaml"
    yaml_path.write_text(yaml_content, encoding="utf-8")
    print(f"\nDataset YAML created: {yaml_path}")
    print("Conversion complete!")


if __name__ == "__main__":
    main()
