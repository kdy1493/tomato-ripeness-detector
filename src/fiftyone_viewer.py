import autorootcwd
import fiftyone as fo
import json
import yaml
from pathlib import Path

def view_results_in_fiftyone():
    """
    Loads both ground truth labels and YOLO prediction results into FiftyOne
    for a side-by-side comparison.
    """
    print("--- Starting FiftyOne Visualization ---")

    # --- 1. 경로 설정 ---
    predictions_path = Path("YOLO_predict_results/yolov10n_baseline_predict/predictions.json")
    gt_path = Path("data/laboro_tomato/annotations/test.json") # Ground Truth
    data_yaml_path = Path("data/laboro_tomato/example_dataset.yaml")

    # --- 필수 파일 확인 ---
    for p in [predictions_path, gt_path, data_yaml_path]:
        if not p.exists():
            print(f"Error: Required file not found at '{p}'")
            return

    # --- 2. FiftyOne 데이터셋 생성 ---
    dataset_name = "smart-farm-evaluation"
    print(f"Creating FiftyOne dataset: '{dataset_name}'")
    if dataset_name in fo.list_datasets():
        fo.delete_dataset(dataset_name)
    dataset = fo.Dataset(dataset_name)

    # --- 3. 클래스 이름 및 데이터 로드 ---
    try:
        with data_yaml_path.open('r') as f:
            class_names = yaml.safe_load(f)['names']
        with predictions_path.open('r') as f:
            predictions_data = json.load(f)
        with gt_path.open('r') as f:
            gt_data = json.load(f)
    except Exception as e:
        print(f"Error loading data files: {e}")
        return

    # --- 4. 데이터 조회를 위한 딕셔너리 준비 ---
    preds_by_stem = {}
    for pred in predictions_data:
        image_id = pred['image_id']
        if image_id not in preds_by_stem:
            preds_by_stem[image_id] = []
        preds_by_stem[image_id].append(pred)
    
    gt_annos_by_img_id = {img['id']: [] for img in gt_data['images']}
    for ann in gt_data['annotations']:
        gt_annos_by_img_id[ann['image_id']].append(ann)

    # --- 5. 데이터셋에 샘플, 정답, 예측 결과 추가 ---
    print("Loading images and adding ground truth & predictions...")
    
    samples = []
    test_image_dir = data_yaml_path.parent / "test" / "images"

    for gt_img_info in gt_data['images']:
        filepath = test_image_dir / gt_img_info['file_name']
        
        if not filepath.exists():
            print(f"Warning: Image '{gt_img_info['file_name']}' not found. Skipping.")
            continue

        sample = fo.Sample(filepath=str(filepath))
        w, h = gt_img_info['width'], gt_img_info['height']

        # A. 정답(Ground Truth) 추가
        gt_detections = []
        for ann in gt_annos_by_img_id.get(gt_img_info['id'], []):
            x, y, box_w, box_h = ann['bbox'] # COCO format: [x, y, width, height]
            gt_detections.append(
                fo.Detection(
                    label=class_names[ann['category_id'] -1],
                    bounding_box=[x/w, y/h, box_w/w, box_h/h],
                )
            )
        sample["ground_truth"] = fo.Detections(detections=gt_detections)

        # B. 예측(Prediction) 추가
        file_stem = filepath.stem
        pred_list = preds_by_stem.get(file_stem, [])
        if pred_list:
            pred_detections = []
            for pred_box in pred_list:
                # COCO prediction format uses 'bbox' and [x, y, width, height]
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

    dataset.add_samples(samples)
    print(f"Successfully loaded {len(samples)} samples.")

    # --- 6. FiftyOne 앱 실행 ---
    print("Launching FiftyOne App. Press Ctrl+C in the terminal to exit.")
    session = fo.launch_app(dataset)
    session.wait()

if __name__ == '__main__':
    view_results_in_fiftyone()