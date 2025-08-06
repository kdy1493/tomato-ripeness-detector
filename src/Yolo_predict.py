import os
import cv2
from ultralytics import YOLO
import yaml

def run_prediction():


    model_path = "YOLO_train_results/yolov10n_baseline/weights/best.pt"
    data_yaml_path = "data/laboro_tomato/example_dataset.yaml"
    output_project = "YOLO_predict_results"
    output_name = "yolov10n_baseline_predict"
    
    model = YOLO(model_path)
    
    print(f"Running validation on the 'val' split specified in '{data_yaml_path}'...")
    val_results = model.val(
        data=data_yaml_path,
        split='val',
        save_json=True,
        project=output_project,
        name=output_name,
        exist_ok=True
    )
    

    results_dir = val_results.save_dir
    json_path = os.path.join(results_dir, "predictions.json")
    print(f"\nValidation complete. Metrics saved in: {results_dir}")
    print(f"Prediction results saved as JSON to: {json_path}")
    
    # --- 4. 예시 이미지 생성 ---
    print("\nGenerating a sample prediction image from the validation set...")
    try:
        with open(data_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
        
        # yaml 파일의 path와 val 경로를 조합하여 전체 경로 생성
        base_path = data_config.get('path', '')
        val_images_rel_path = data_config['val']
        val_image_dir = os.path.join(base_path, val_images_rel_path)
        
        # 검증셋의 첫 번째 이미지 선택
        sample_image_name = os.listdir(val_image_dir)[0]
        sample_image_path = os.path.join(val_image_dir, sample_image_name)
        
        # 선택된 이미지로 예측 실행
        print(f"Running prediction on sample image: {sample_image_path}")
        sample_prediction_results = model(sample_image_path)
        
        # 결과 시각화 및 저장
        annotated_frame = sample_prediction_results[0].plot(line_width=2, font_size=15)
        sample_output_path = os.path.join(results_dir, 'validation_sample_prediction.jpg')
        cv2.imwrite(sample_output_path, annotated_frame)
        
        print(f"Sample prediction image saved to: {sample_output_path}")
        
    except (FileNotFoundError, IndexError, KeyError) as e:
        print(f"\nCould not generate sample image. Error: {e}")
        print("Please ensure the dataset YAML file and validation image directory are correct and not empty.")

if __name__ == '__main__':
    run_prediction()

