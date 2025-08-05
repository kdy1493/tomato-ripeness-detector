import torch
from ultralytics import YOLO

def main():
    # CUDA 사용 가능 여부 및 현재 장치 확인
    if torch.cuda.is_available():
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        device = 0
    else:
        print("CUDA is not available. Using CPU.")
        device = 'cpu'

    # 사전 학습된 YOLO 모델을 로드
    model = YOLO('yolov10n.pt')

    # 커스텀 데이터셋으로 모델을 학습
    results = model.train(
        data='data/laboro_tomato/example_dataset.yaml',
        epochs=100,
        batch=-1,  # VRAM에 맞춰 자동 조절
        imgsz=640,
        patience=50,
        optimizer='auto',
        project='train_results',
        name='yolov10n_baseline',
        exist_ok=True,
        device=device
    )

    print("Model training completed.")
    print(f"Results saved to {results.save_dir}")

if __name__ == '__main__':
    main()
