import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class CocoImage:
    id: int
    file_name: str
    width: int
    height: int


@dataclass
class CocoAnnotation:
    id: int
    image_id: int
    category_id: int
    bbox: List[float]  # [x, y, w, h]


def load_coco(json_path: Path) -> Tuple[Dict[int, CocoImage], Dict[int, List[CocoAnnotation]], Dict[int, str]]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    images: Dict[int, CocoImage] = {}
    for im in data.get("images", []):
        images[int(im["id"])] = CocoImage(
            id=int(im["id"]),
            file_name=str(im["file_name"]),
            width=int(im["width"]),
            height=int(im["height"]),
        )

    anns_by_image: Dict[int, List[CocoAnnotation]] = {}
    for ann in data.get("annotations", []):
        ann_obj = CocoAnnotation(
            id=int(ann["id"]),
            image_id=int(ann["image_id"]),
            category_id=int(ann["category_id"]),
            bbox=[float(x) for x in ann["bbox"]],
        )
        anns_by_image.setdefault(ann_obj.image_id, []).append(ann_obj)

    categories: Dict[int, str] = {}
    for cat in data.get("categories", []):
        categories[int(cat["id"])] = str(cat["name"])  # e.g., 1->unripe, 2->semi-ripe, 3->fully-ripe

    return images, anns_by_image, categories


def coco_bbox_to_yolo(bbox_xywh: List[float], img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    x, y, w, h = bbox_xywh
    cx = (x + w / 2.0) / img_w
    cy = (y + h / 2.0) / img_h
    nw = w / img_w
    nh = h / img_h
    return cx, cy, nw, nh


def write_yolo_label(label_path: Path, records: List[Tuple[int, float, float, float, float]]):
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with label_path.open("w", encoding="utf-8") as f:
        for cls, cx, cy, w, h in records:
            f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def copy_image(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def convert_split(
    images: Dict[int, CocoImage],
    anns_by_image: Dict[int, List[CocoAnnotation]],
    catid_to_name: Dict[int, str],
    images_dir: Path,
    out_split_dir: Path,
    image_ids: List[int],
    catid_to_yolo: Dict[int, int],
):
    for image_id in image_ids:
        if image_id not in images:
            continue
        im = images[image_id]
        src_img = images_dir / im.file_name
        dst_img = out_split_dir / "images" / im.file_name

        # copy image
        if not src_img.exists():
            # Try nested paths if COCO file_name had subdirs; otherwise warn and skip
            raise FileNotFoundError(f"Image not found: {src_img}")
        copy_image(src_img, dst_img)

        # prepare labels
        yolo_records: List[Tuple[int, float, float, float, float]] = []
        for ann in anns_by_image.get(image_id, []):
            if ann.category_id not in catid_to_yolo:
                # skip unknown categories
                continue
            cls_idx = catid_to_yolo[ann.category_id]
            cx, cy, w, h = coco_bbox_to_yolo(ann.bbox, im.width, im.height)
            # clamp to [0,1]
            cx = min(max(cx, 0.0), 1.0)
            cy = min(max(cy, 0.0), 1.0)
            w = min(max(w, 0.0), 1.0)
            h = min(max(h, 0.0), 1.0)
            if w <= 0.0 or h <= 0.0:
                continue
            yolo_records.append((cls_idx, cx, cy, w, h))

        label_name = Path(im.file_name).with_suffix(".txt").name
        label_path = out_split_dir / "labels" / label_name
        write_yolo_label(label_path, yolo_records)


def main():
    """
    Convert tomatOD COCO annotations to YOLO format, and create a train/val/test folder
    structure under data/tomatOD suitable for Ultralytics training.

    Assumes the following input layout:
      - data/tomatOD/tomatOD_annotations/tomatOD_train.json
      - data/tomatOD/tomatOD_annotations/tomatOD_test.json
      - data/tomatOD/tomatOD_images/train
      - data/tomatOD/tomatOD_images/test
    """

    project_root = Path(__file__).resolve().parents[1]
    data_root = project_root / "data" / "tomatOD"

    coco_train_json = data_root / "tomatOD_annotations" / "tomatOD_train.json"
    coco_test_json = data_root / "tomatOD_annotations" / "tomatOD_test.json"
    images_train_dir = data_root / "tomatOD_images" / "train"
    images_test_dir = data_root / "tomatOD_images" / "test"

    out_root = data_root
    out_train = out_root / "train"
    out_val = out_root / "val"
    out_test = out_root / "test"

    val_ratio = 0.1
    seed = 777
    random.seed(seed)

    # Load train COCO
    train_images, train_anns_by_image, catid_to_name = load_coco(coco_train_json)

    # Map category ids (1-based in COCO) -> YOLO class indices (0-based)
    sorted_cat_ids = sorted(catid_to_name.keys())
    catid_to_yolo = {cat_id: idx for idx, cat_id in enumerate(sorted_cat_ids)}

    # Split train into train/val by image ids
    all_train_image_ids = list(train_images.keys())
    random.shuffle(all_train_image_ids)
    val_count = max(1, int(len(all_train_image_ids) * val_ratio))
    val_image_ids = set(all_train_image_ids[:val_count])
    real_train_image_ids = [i for i in all_train_image_ids if i not in val_image_ids]

    # Convert train split
    convert_split(
        images=train_images,
        anns_by_image=train_anns_by_image,
        catid_to_name=catid_to_name,
        images_dir=images_train_dir,
        out_split_dir=out_train,
        image_ids=real_train_image_ids,
        catid_to_yolo=catid_to_yolo,
    )

    # Convert val split
    convert_split(
        images=train_images,
        anns_by_image=train_anns_by_image,
        catid_to_name=catid_to_name,
        images_dir=images_train_dir,
        out_split_dir=out_val,
        image_ids=list(val_image_ids),
        catid_to_yolo=catid_to_yolo,
    )

    # Convert test split from test COCO
    test_images, test_anns_by_image, _ = load_coco(coco_test_json)
    convert_split(
        images=test_images,
        anns_by_image=test_anns_by_image,
        catid_to_name=catid_to_name,
        images_dir=images_test_dir,
        out_split_dir=out_test,
        image_ids=list(test_images.keys()),
        catid_to_yolo=catid_to_yolo,
    )

    # Write dataset YAML for Ultralytics
    yaml_path = out_root / "example_dataset.yaml"
    yaml_text = (
        "# Ultralytics dataset YAML for tomatOD (converted from COCO to YOLO)\n"
        f"path: {out_root.as_posix()}\n"
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n"
        f"nc: {len(sorted_cat_ids)}\n"
        f"names: [{', '.join(catid_to_name[cid] for cid in sorted_cat_ids)}]\n"
    )
    yaml_path.write_text(yaml_text, encoding="utf-8")

    print("Conversion complete.")
    print(f"- Train images: {len(real_train_image_ids)} -> {out_train}")
    print(f"- Val images: {len(val_image_ids)} -> {out_val}")
    print(f"- Test images: {len(test_images)} -> {out_test}")
    print(f"Dataset YAML: {yaml_path}")


if __name__ == "__main__":
    main()


