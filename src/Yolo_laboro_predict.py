import autorootcwd
import os
import cv2
from ultralytics import YOLO
import yaml
from pathlib import Path

def run_prediction():

    model_path = "YOLO_laboro_train_results/yolov10n_baseline/weights/best.pt"
    data_yaml_path = Path("data/laboro_tomato/example_dataset.yaml")
    output_project = "YOLO_laboro_predict_results"
    output_name = "yolov10n_baseline_predict"
    
    model = YOLO(model_path)
    
    print(f"Running prediction on the 'test' split specified in '{data_yaml_path}'...")
    val_results = model.val(
        data=str(data_yaml_path),
        split='test',  # Use the 'test' split for final evaluation
        save_json=True,
        project=output_project,
        name=output_name,
        exist_ok=True
    )
    
    results_dir = Path(val_results.save_dir)
    json_path = results_dir / "predictions.json"
    print(f"\nPrediction complete. Metrics saved in: {results_dir}")
    print(f"Prediction results saved as JSON to: {json_path}")
    
    # --- 4. 예시 이미지 생성 ---
    print("\nGenerating a sample prediction image from the test set...")
    try:
        with data_yaml_path.open('r') as f:
            data_config = yaml.safe_load(f)
        
        # Get the directory of the YAML file itself to resolve relative paths
        yaml_dir = data_yaml_path.parent
        test_images_rel_path = data_config['test']
        test_image_dir = yaml_dir / test_images_rel_path
        
        # Get the first image from the test directory
        image_files = [f for f in test_image_dir.iterdir() if f.is_file()]
        if not image_files:
            raise IndexError("No images found in the test directory.")
            
        sample_image_path = image_files[0]
        
        # Run prediction on the sample image
        print(f"Running prediction on sample image: {sample_image_path}")
        sample_prediction_results = model(str(sample_image_path))
        
        # Plot the results and save the image
        annotated_frame = sample_prediction_results[0].plot(line_width=2, font_size=15)
        sample_output_path = results_dir / 'test_sample_prediction.jpg'
        cv2.imwrite(str(sample_output_path), annotated_frame)
        
        print(f"Sample prediction image saved to: {sample_output_path}")
        
    except (FileNotFoundError, IndexError, KeyError) as e:
        print(f"\nCould not generate sample image. Error: {e}")
        print("Please ensure the dataset YAML file and test image directory are correct and not empty.")

if __name__ == '__main__':
    run_prediction()
