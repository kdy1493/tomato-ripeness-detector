import torch
from ultralytics import YOLO
from pathlib import Path
import shutil


def main():
    if torch.cuda.is_available():
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        device = 0
    else:
        print("CUDA is not available. Using CPU.")
        device = 'cpu'

    # 동일한 모델/설정으로 laboro 스크립트와 맞춤
    model = YOLO('yolov10n.pt')

    # 결과 폴더 초기화 (존재하면 삭제 후 재생성)
    project = 'YOLO_tomatOD_train_results'
    name = 'yolov10n_tomatOD'
    output_dir = Path(project) / name
    if output_dir.exists():
        print(f"Existing directory found. Removing: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = model.train(
        data='data/tomatOD/example_dataset.yaml',
        epochs=100,
        batch=-1,
        imgsz=640,
        patience=50,
        project=project,
        name=name,
        exist_ok=True,
        device=device,

        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.1,
        fliplr=0.5,

        cos_lr=True,
        optimizer='AdamW'
    )

    print("Model training completed.")
    print(f"Results saved to {results.save_dir}")


if __name__ == '__main__':
    main()


