"""
Convert custom_tomato_data_coco (COCO format) to custom_tomato_data_yolo (YOLO format).

Input:
  data/custom_tomato_dataset/custom_tomato_data_coco/
    _annotations.coco.json
    images/*.jpg

Output:
  data/custom_tomato_dataset/custom_tomato_data_yolo/
    dataset.yaml
    images/train/*.jpg
    labels/train/*.txt

Categories (COCO id -> YOLO class):
  0 ripe   -> 0
  1 unripe -> 1
"""

import json
import shutil
from pathlib import Path


def coco_bbox_to_yolo(bbox, img_w: int, img_h: int):
    """Convert COCO [x, y, w, h] (absolute) to YOLO [cx, cy, w, h] (normalized)."""
    x, y, w, h = [float(v) for v in bbox]
    cx = (x + w / 2.0) / img_w
    cy = (y + h / 2.0) / img_h
    nw = w / img_w
    nh = h / img_h
    return (
        max(0.0, min(1.0, cx)),
        max(0.0, min(1.0, cy)),
        max(0.0, min(1.0, nw)),
        max(0.0, min(1.0, nh)),
    )


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    base = project_root / "data" / "custom_tomato_dataset"
    src_dir = base / "custom_tomato_data_coco"
    dst_dir = base / "custom_tomato_data_yolo"

    src_json = src_dir / "_annotations.coco.json"
    src_images = src_dir / "images"

    with src_json.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    categories = sorted(coco["categories"], key=lambda c: c["id"])
    cat_id_to_yolo = {cat["id"]: idx for idx, cat in enumerate(categories)}
    class_names = [cat["name"] for cat in categories]

    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"COCO cat_id -> YOLO class: {cat_id_to_yolo}")

    dst_images_dir = dst_dir / "images" / "train"
    dst_labels_dir = dst_dir / "labels" / "train"
    dst_images_dir.mkdir(parents=True, exist_ok=True)
    dst_labels_dir.mkdir(parents=True, exist_ok=True)

    images_by_id = {img["id"]: img for img in coco["images"]}
    anns_by_image: dict[int, list] = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    converted = 0
    skipped_imgs = 0
    total_labels = 0

    for img_id, img_info in images_by_id.items():
        file_name = img_info["file_name"]
        src_img = src_images / file_name
        if not src_img.exists():
            skipped_imgs += 1
            continue

        shutil.copy2(src_img, dst_images_dir / file_name)

        img_w = img_info["width"]
        img_h = img_info["height"]

        label_lines = []
        for ann in anns_by_image.get(img_id, []):
            cat_id = ann["category_id"]
            if cat_id not in cat_id_to_yolo:
                continue
            yolo_cls = cat_id_to_yolo[cat_id]
            cx, cy, nw, nh = coco_bbox_to_yolo(ann["bbox"], img_w, img_h)
            if nw <= 0 or nh <= 0:
                continue
            label_lines.append(f"{yolo_cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        label_path = dst_labels_dir / (Path(file_name).stem + ".txt")
        label_path.write_text("\n".join(label_lines), encoding="utf-8")
        converted += 1
        total_labels += len(label_lines)

    yaml_content = f"""\
path: {dst_dir.as_posix()}
train: images/train
val: images/train

nc: {len(class_names)}
names: {class_names}
"""
    yaml_path = dst_dir / "dataset.yaml"
    yaml_path.write_text(yaml_content, encoding="utf-8")

    print(f"\nConverted {converted} images, skipped {skipped_imgs}")
    print(f"Total label entries: {total_labels}")
    print(f"Images  -> {dst_images_dir}")
    print(f"Labels  -> {dst_labels_dir}")
    print(f"YAML    -> {yaml_path}")


if __name__ == "__main__":
    main()
