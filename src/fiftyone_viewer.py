import autorootcwd
import fiftyone as fo

dataset_name = "final_test_evaluation"
dataset_dir = "data/laboro_tomato"
predictions_path = "YOLO_predict_results/yolov10n_baseline_predict/predictions.json"

# dataset reset 
if fo.dataset_exists(dataset_name):
    fo.delete_dataset(dataset_name)

print("Creating FiftyOne dataset from the 'test' data source...")
dataset = fo.Dataset.from_dir(
    dataset_dir=dataset_dir,
    dataset_type=fo.types.COCODetectionDataset,
    name=dataset_name,
    data_path="test/images",
    labels_path="test/labels",
    label_field="ground_truth"
)

print(f"Adding YOLO predictions from '{predictions_path}'...")
fo.add_coco_predictions(
    dataset,
    predictions_path=predictions_path,
    eval_key="predictions"
)

session = fo.launch_app(dataset, address="127.0.0.1")
print(f"FiftyOne App is running at: {session.url}")

session.wait()
