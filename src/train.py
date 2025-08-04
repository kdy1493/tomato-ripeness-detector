from ultralytics import YOLO

def main():
    # 사전 학습된 YOLO 모델을 로드
    model = YOLO('yolov10n.pt')

    # 커스텀 데이터셋으로 모델을 학습
    results = model.train(
        data='data/laboro-tomato/example_dataset.yaml',
        epochs=100,
        imgsz=640,
        project='train-results',
        name='yolov10n_tomato_custom',
        device=0
    )

    print("Model training completed.")
    print(f"Results saved to {results.save_dir}")

if __name__ == '__main__':
    main()
