from ultralytics import YOLO
import cv2

def main():
    # Load the best performing model
    model_path = 'runs/yolov10n_tomato_custom/weights/best.pt'
    model = YOLO(model_path)

    # Image to be used for inference
    image_path = 'data/tomato_detection/images/tomato0.png'
    
    # Perform inference
    results = model(image_path)

    # Visualize the results
    annotated_frame = results[0].plot()

    # Save the result image
    output_path = 'prediction_result.jpg'
    cv2.imwrite(output_path, annotated_frame)

    print(f"Prediction result saved to {output_path}")

if __name__ == '__main__':
    main()
