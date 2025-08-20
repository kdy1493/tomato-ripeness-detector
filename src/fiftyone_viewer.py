import autorootcwd
import fiftyone as fo
import json
import yaml
from pathlib import Path
import argparse
from typing import Optional

def visualize_in_fiftyone(
    gt_path: Path,
    data_yaml_path: Path,
    split: str,
    predictions_path: Optional[Path] = None,
    dataset_name: Optional[str] = None,
):
    """
    Loads dataset splits into FiftyOne for exploration or prediction comparison.

    - If predictions_path is provided, it loads predictions for comparison (evaluation mode).
    - Otherwise, it loads only the ground truth for exploration (exploration mode).

    Args:
        gt_path (Path): Path to the ground truth COCO JSON file for the split.
        data_yaml_path (Path): Path to the dataset's YAML configuration file.
        split (str): The dataset split to load (e.g., 'train', 'val', 'test').
        predictions_path (Optional[Path]): Path to the YOLO predictions JSON file.
        dataset_name (Optional[str]): The name for the FiftyOne dataset.
    """
    mode = "Evaluation" if predictions_path else "Exploration"
    print(f"--- Starting FiftyOne {mode} for '{split}' split ---")

    # --- 1. 필수 파일 확인 ---
    required_paths = [gt_path, data_yaml_path]
    if predictions_path:
        required_paths.append(predictions_path)

    for p in required_paths:
        if not p.exists():
            print(f"Error: Required file not found at '{p}'")
            return

    # --- 2. FiftyOne 데이터셋 이름 및 생성 ---
    if not dataset_name:
        yaml_stem = data_yaml_path.stem.replace("example_", "")
        mode_str = "eval" if predictions_path else "explore"
        dataset_name = f"{yaml_stem}-{split}-{mode_str}"

    print(f"Creating FiftyOne dataset: '{dataset_name}'")
    if dataset_name in fo.list_datasets():
        fo.delete_dataset(dataset_name)
    dataset = fo.Dataset(dataset_name)

    # --- 3. 클래스 이름 및 데이터 로드 ---
    try:
        with data_yaml_path.open('r') as f:
            data_yaml = yaml.safe_load(f)
            class_names = data_yaml['names']
            image_dir_rel = data_yaml.get(split)
            if not image_dir_rel:
                print(f"Error: Split '{split}' not found in '{data_yaml_path}'")
                return

        with gt_path.open('r') as f:
            gt_data = json.load(f)

        predictions_data = None
        if predictions_path:
            with predictions_path.open('r') as f:
                predictions_data = json.load(f)

    except Exception as e:
        print(f"Error loading data files: {e}")
        return

    image_dir = data_yaml_path.parent / image_dir_rel

    # --- 4. 데이터 조회를 위한 딕셔너리 준비 ---
    preds_by_stem = {}
    if predictions_data:
        for pred in predictions_data:
            image_id = pred['image_id']
            if image_id not in preds_by_stem:
                preds_by_stem[image_id] = []
            preds_by_stem[image_id].append(pred)

    gt_annos_by_img_id = {img['id']: [] for img in gt_data['images']}
    for ann in gt_data['annotations']:
        gt_annos_by_img_id[ann['image_id']].append(ann)

    # --- 5. 데이터셋에 샘플 추가 ---
    print(f"Loading images and labels from '{split}' split...")

    samples = []
    for gt_img_info in gt_data['images']:
        filepath = image_dir / gt_img_info['file_name']

        if not filepath.exists():
            continue

        sample = fo.Sample(filepath=str(filepath))
        w, h = gt_img_info['width'], gt_img_info['height']

        # A. 정답(Ground Truth) 추가
        gt_detections = []
        for ann in gt_annos_by_img_id.get(gt_img_info['id'], []):
            x, y, box_w, box_h = ann['bbox']
            gt_detections.append(
                fo.Detection(
                    label=class_names[ann['category_id'] - 1],
                    bounding_box=[x/w, y/h, box_w/w, box_h/h],
                )
            )
        sample["ground_truth"] = fo.Detections(detections=gt_detections)

        # B. 예측(Prediction) 추가 (평가 모드인 경우)
        if predictions_data:
            file_stem = filepath.stem
            pred_list = preds_by_stem.get(file_stem, [])
            if pred_list:
                pred_detections = []
                for pred_box in pred_list:
                    x, y, box_w, box_h = pred_box['bbox']
                    pred_detections.append(
                        fo.Detection(
                            label=class_names[pred_box['category_id'] - 1],
                            bounding_box=[x/w, y/h, box_w/w, box_h/h],
                            confidence=pred_box['score']
                        )
                    )
                sample["predictions"] = fo.Detections(detections=pred_detections)

        samples.append(sample)
    
    if not samples:
        print(f"No images found for split '{split}' in directory '{image_dir}'.")
        print("Please check your COCO JSON and dataset YAML file paths.")
        return

    dataset.add_samples(samples)
    print(f"Successfully loaded {len(samples)} samples.")

    # --- 6. FiftyOne 앱 실행 ---
    print("Launching FiftyOne App. Press Ctrl+C in the terminal to exit.")
    session = fo.launch_app(dataset, auto=False)
    session.wait()

def main():
    parser = argparse.ArgumentParser(
        description="Visualize YOLO datasets and predictions in FiftyOne."
    )
    parser.add_argument(
        "--gt",
        type=Path,
        required=True,
        help="Path to the ground truth COCO JSON file.",
    )
    parser.add_argument(
        "--dataset-yaml",
        type=Path,
        required=True,
        help="Path to the dataset's YAML configuration file.",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="(Evaluation mode) Path to the YOLO predictions JSON file. If not provided, runs in exploration mode.",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=['train', 'val', 'test'],
        default='val',
        help="The dataset split to load (train, val, or test). Defaults to 'val'.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Optional name for the FiftyOne dataset. Auto-generated if not provided.",
    )
    args = parser.parse_args()

    visualize_in_fiftyone(
        gt_path=args.gt,
        data_yaml_path=args.dataset_yaml,
        split=args.split,
        predictions_path=args.predictions,
        dataset_name=args.dataset_name,
    )

if __name__ == '__main__':
    main()
