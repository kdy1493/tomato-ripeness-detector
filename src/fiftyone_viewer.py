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
    # --- CVAT options ---
    to_cvat: bool = False,
    from_cvat: bool = False,
    anno_key: Optional[str] = None,
    cvat_url: Optional[str] = None,
    launch_editor: bool = True,
    segment_size: Optional[int] = None,
    assignee: Optional[str] = None,
    cvat_cleanup: bool = False,
    limit: Optional[int] = None,
):
    """
    Loads dataset splits into FiftyOne for exploration or prediction comparison.
    Optionally integrates with CVAT to upload (annotate) and then load results.

    Args:
        gt_path (Path): Path to the ground truth COCO JSON file for the split.
        data_yaml_path (Path): Path to the dataset's YAML configuration file.
        split (str): The dataset split to load (e.g., 'train', 'val', 'test').
        predictions_path (Optional[Path]): Path to the YOLO predictions JSON file.
        dataset_name (Optional[str]): The name for the FiftyOne dataset.

        to_cvat (bool): If True, push current view to CVAT for labeling.
        from_cvat (bool): If True, load finished labels from CVAT with the same anno_key.
        anno_key (str): Annotation run key (unique identifier for this round-trip).
        cvat_url (str): CVAT server URL (e.g., http://localhost:8080). Optional if using cloud+env.
        launch_editor (bool): Open CVAT editor after upload.
        segment_size (int): Split the task into jobs of this size (for large datasets).
        assignee (str): Assign CVAT task to this username (optional).
        cvat_cleanup (bool): If True, clean up backend metadata after loading results.
        limit (int): An optional maximum number of samples to load.
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
        with data_yaml_path.open('r', encoding='utf-8') as f:
            data_yaml = yaml.safe_load(f)
            class_names = data_yaml['names']
            image_dir_rel = data_yaml.get(split)
            if not image_dir_rel:
                print(f"Error: Split '{split}' not found in '{data_yaml_path}'")
                return

        with gt_path.open('r', encoding='utf-8') as f:
            gt_data = json.load(f)

        predictions_data = None
        if predictions_path:
            with predictions_path.open('r', encoding='utf-8') as f:
                predictions_data = json.load(f)

    except Exception as e:
        print(f"Error loading data files: {e}")
        return

    image_dir = data_yaml_path.parent / image_dir_rel

    # --- 4. 데이터 조회를 위한 딕셔너리 준비 ---
    preds_by_stem = {}
    if predictions_data:
        # NOTE: pred['image_id'] 대신 파일 stem으로 매칭하려면 예측 JSON 구조를 맞춰야 함
        for pred in predictions_data:
            image_id = pred.get('image_id')
            if image_id is None:
                # 예: {"file_stem": "000001", ...} 형태라면 pred['file_stem']으로 교체
                continue
            preds_by_stem.setdefault(image_id, []).append(pred)

    gt_annos_by_img_id = {img['id']: [] for img in gt_data['images']}
    for ann in gt_data['annotations']:
        gt_annos_by_img_id[ann['image_id']].append(ann)

    # --- 5. 데이터셋에 샘플 추가 ---
    print(f"Loading images and labels from '{split}' split...")

    samples = []
    loaded_count = 0
    for gt_img_info in gt_data['images']:
        filepath = image_dir / gt_img_info['file_name']
        if not filepath.exists():
            continue

        sample = fo.Sample(filepath=str(filepath))
        w, h = gt_img_info['width'], gt_img_info['height']

        # A. 정답(GT)
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

        # B. 예측(옵션)
        if predictions_data:
            # COCO-style predictions가 image_id(int)만 주는 경우, 파일 stem 기반 매칭이 안 맞을 수 있음
            # 필요하면 예측 생성 시 file_stem을 함께 기록해두는 걸 권장
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
                            confidence=pred_box.get('score'),
                        )
                    )
                sample["predictions"] = fo.Detections(detections=pred_detections)

        samples.append(sample)
        loaded_count += 1
        if limit and loaded_count >= limit:
            print(f"Reached sample limit ({limit}).")
            break

    if not samples:
        print(f"No images found for split '{split}' in directory '{image_dir}'.")
        print("Please check your COCO JSON and dataset YAML file paths.")
        return

    dataset.add_samples(samples)
    print(f"Successfully loaded {len(samples)} samples.")

    # ----------------------
    # CVAT INTEGRATION HERE
    # ----------------------
    view = dataset.view()  # 필요 시 .match(..) 등으로 일부만 보낼 수 있음

    if to_cvat:
        if not anno_key:
            print("Error: --to-cvat 사용 시 --anno-key 는 필수입니다.")
            return
        cvat_kwargs = dict(
            backend="cvat",
            label_field="ground_truth",   # CVAT에서 편집할 필드(없으면 생성)
            launch_editor=launch_editor,
        )
        if cvat_url:
            cvat_kwargs["url"] = cvat_url
        if segment_size:
            cvat_kwargs["segment_size"] = int(segment_size)
        if assignee:
            # 지원하는 환경이면 담당자 지정
            cvat_kwargs["task_assignee"] = assignee

        print(f"[CVAT] Uploading view to CVAT with anno_key='{anno_key}' ...")
        view.annotate(anno_key, **cvat_kwargs)
        print("[CVAT] Task/Jobs created. Do labeling in CVAT, then run with --from-cvat to fetch results.")

    if from_cvat:
        if not anno_key:
            print("Error: --from-cvat 사용 시 --anno-key 는 필수입니다.")
            return
        print(f"[CVAT] Loading annotations back from CVAT (anno_key='{anno_key}') ...")
        dataset.load_annotations(anno_key, cleanup=bool(cvat_cleanup))
        print("[CVAT] Loaded annotations and merged into dataset.")

    # --- 6. FiftyOne 앱 실행 ---
    print("Launching FiftyOne App. Press Ctrl+C in the terminal to exit.")
    session = fo.launch_app(dataset, auto=False)
    session.wait()

def main():
    # 설정 파일을 먼저 파싱하여 기본값으로 사용
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "-c", "--config", type=Path,
        help="YAML 설정 파일 경로. CLI 인자가 설정 파일보다 우선 적용됩니다."
    )
    config_args, remaining_argv = config_parser.parse_known_args()

    config_from_file = {}
    if config_args.config and config_args.config.exists():
        with config_args.config.open('r', encoding='utf-8') as f:
            config_from_file = yaml.safe_load(f)

    # 모든 인자를 포함하는 메인 파서
    parser = argparse.ArgumentParser(
        description="FiftyOne을 사용하여 YOLO 데이터셋과 예측 시각화 (+ CVAT 연동).",
        parents=[config_parser]  # --config 인자 상속
    )

    parser.add_argument("--gt", type=Path, help="Ground truth COCO JSON 파일 경로.")
    parser.add_argument("--dataset-yaml", type=Path, help="데이터셋의 YAML 설정 파일 경로.")
    parser.add_argument("--predictions", type=Path, help="(평가 모드) Predictions JSON 파일 경로.")
    parser.add_argument("--split", type=str, choices=['train', 'val', 'test'], help="로드할 데이터 스플릿.")
    parser.add_argument("--dataset-name", type=str, help="사용자 정의 FiftyOne 데이터셋 이름.")
    parser.add_argument("--limit", type=int, help="로드할 최대 샘플 개수 (테스트용).")

    # --- CVAT 플래그 ---
    parser.add_argument("--to-cvat", action="store_true", help="현재 뷰를 라벨링을 위해 CVAT에 업로드합니다.")
    parser.add_argument("--from-cvat", action="store_true", help="동일한 anno key로 CVAT에서 완료된 라벨을 로드합니다.")
    parser.add_argument("--anno-key", type=str, help="Annotation 실행 키 (CVAT 작업에 필수).")
    parser.add_argument("--cvat-url", type=str, help="CVAT 서버 URL (예: http://localhost:8080).")
    parser.add_argument("--launch-editor", action="store_true", help="업로드 후 CVAT 에디터를 엽니다.")
    parser.add_argument("--segment-size", type=int, help="이 개수만큼의 샘플로 작업을 나눕니다.")
    parser.add_argument("--assignee", type=str, help="이 사용자 이름으로 CVAT 작업을 할당합니다.")
    parser.add_argument("--cvat-cleanup", action="store_true", help="결과 로드 시 백엔드 메타데이터를 정리합니다.")

    # 설정 파일 값으로 기본값 설정 후, 나머지 인자 파싱
    parser.set_defaults(**config_from_file)
    args = parser.parse_args(remaining_argv)

    # --- 필수 인자 확인 ---
    if not args.gt or not args.dataset_yaml:
        print("오류: --gt 와 --dataset-yaml 은 필수 인자입니다.")
        print("커맨드 라인 또는 --config YAML 파일을 통해 제공해야 합니다.")
        return

    # --split 자동 유추
    split = args.split
    if not split:
        gt_filename = Path(args.gt).stem.lower()
        if "train" in gt_filename:
            split = "train"
        elif "val" in gt_filename:
            split = "val"
        elif "test" in gt_filename:
            split = "test"
        else:
            print("오류: --gt 파일명에서 스플릿을 유추할 수 없습니다.")
            print("--split {train,val,test} 로 명시해주세요.")
            return

    # YAML에서 로드된 경로가 문자열일 수 있으므로 Path 객체로 변환
    gt_path = Path(args.gt)
    data_yaml_path = Path(args.dataset_yaml)
    predictions_path = Path(args.predictions) if args.predictions else None

    visualize_in_fiftyone(
        gt_path=gt_path,
        data_yaml_path=data_yaml_path,
        split=split,
        predictions_path=predictions_path,
        dataset_name=args.dataset_name,
        # CVAT
        to_cvat=args.to_cvat,
        from_cvat=args.from_cvat,
        anno_key=args.anno_key,
        cvat_url=args.cvat_url,
        launch_editor=args.launch_editor,
        segment_size=args.segment_size,
        assignee=args.assignee,
        cvat_cleanup=args.cvat_cleanup,
        limit=args.limit,
    )

if __name__ == '__main__':
    main()
