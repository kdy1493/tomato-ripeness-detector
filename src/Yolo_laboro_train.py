import torch
from ultralytics import YOLO

def main():
    if torch.cuda.is_available():
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        device = 0
    else:
        print("CUDA is not available. Using CPU.")
        device = 'cpu'

    model = YOLO('yolov10n.pt')

    # --- 모델 학습 (데이터 증강 및 고급 옵션 추가) ---
    results = model.train(
        # --- 기본 설정 ---
        data='data/laboro_tomato/example_dataset.yaml',
        epochs=100,
        batch=-1,
        imgsz=640,
        patience=50,
        project='YOLO_train_results',
        name='yolov10n_augmented', # 새로운 실험 이름
        exist_ok=True,
        device=device,
        
        # --- 성능 향상을 위한 추가 옵션 ---
        # 1. 데이터 증강 활성화
        augment=True,
        hsv_h=0.015,     # 색상(H) 변형 강도
        hsv_s=0.7,       # 채도(S) 변형 강도
        hsv_v=0.4,       # 명도(V) 변형 강도
        degrees=10.0,    # 회전 각도 범위
        translate=0.1,   # 이동 비율
        scale=0.1,       # 크기 조절 비율
        fliplr=0.5,      # 좌우 반전 확률
        
        # 2. 학습률 스케줄러 변경
        cos_lr=True,     # 코사인 학습률 스케줄러 사용
        
        # 3. 옵티마이저 (기본값 'auto'도 좋지만 명시적으로 설정 가능)
        optimizer='AdamW'
    )

    print("Model training completed.")
    print(f"Results saved to {results.save_dir}")

if __name__ == '__main__':
    main()
