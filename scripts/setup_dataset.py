import autorootcwd  # Sets the CWD to the project root
import kagglehub
import random
import json
import shutil
from pathlib import Path

def download_and_move_dataset():
    print("--- 1. Starting Dataset Download ---")
    dest_dir = Path('data/laboro_tomato')
    
    # If the destination directory already has content, skip download and move.
    if dest_dir.exists() and any(dest_dir.iterdir()):
        print(f"Dataset directory '{dest_dir}' is not empty. Assuming it's already set up.")
        return True

    try:
        print("Downloading 'nexuswho/laboro-tomato' from Kaggle Hub...")
        source_path = Path(kagglehub.dataset_download("nexuswho/laboro-tomato"))
        print(f"Dataset downloaded to cache: {source_path}")

        dest_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Moving dataset files from cache to '{dest_dir}'...")
        for item in source_path.iterdir():
            print(f"  - Moving {item.name}")
            item.rename(dest_dir / item.name)
            
        print("Dataset moved successfully.")
        return True

    except Exception as e:
        print(f"\nAn error occurred during download or move: {e}")
        return False


# data/laboro_tomato/val/images의 50%를 data/laboro_tomato/test/images로 이동하고,
# data/laboro_tomato/val/labels의 50%를 data/laboro_tomato/test/labels로 이동하며,
# data/laboro_tomato/annotations/test.json의 50% annotation을 data/laboro_tomato/annotations/val.json으로 분리하는 함수
def split_dataset_files():
    """Splits the validation set into new val and test sets (50/50 split)."""
    print("\n--- 2. Preparing and Splitting Dataset ---")
    
    base_dir = Path('data/laboro_tomato')
    val_img_dir = base_dir / 'val' / 'images'
    test_img_dir = base_dir / 'test' / 'images'

    if test_img_dir.exists() and any(test_img_dir.iterdir()):
        print("'test' directory already exists and is not empty. Skipping split.")
        return True

    # Final robustness check: ensure val directory exists and is not empty
    if not val_img_dir.exists() or not any(val_img_dir.iterdir()):
        print(f"Validation directory '{val_img_dir}' is empty or does not exist. Cannot split.")
        return False

    val_lbl_dir = base_dir / 'val/labels'
    test_lbl_dir = base_dir / 'test/labels'
    annotations_dir = base_dir / 'annotations'
    
    test_img_dir.mkdir(parents=True, exist_ok=True)
    test_lbl_dir.mkdir(parents=True, exist_ok=True)

    val_images = [f.name for f in val_img_dir.iterdir() if f.is_file()]
    random.shuffle(val_images)
    files_to_move = val_images[:len(val_images) // 2]
    
    if not files_to_move:
        print("Not enough files in validation set to create a test split.")
        return False

    print(f"Moving {len(files_to_move)} files from 'val' to 'test' directories.")
    for filename in files_to_move:
        base = Path(filename).stem
        (val_img_dir / filename).rename(test_img_dir / filename)
        
        label_file = val_lbl_dir / f"{base}.txt"
        if label_file.exists():
            label_file.rename(test_lbl_dir / label_file.name)

    original_val_json = annotations_dir / 'test.json'
    if original_val_json.exists():
        split_annotations(original_val_json, files_to_move, annotations_dir / 'test.json', annotations_dir / 'val.json')
    
    print("--- Dataset Preparation Complete ---")
    return True


# data/annotation/test.json 을 기준으로 50%를 test.json, 50%를 val.json 으로 나누는 함수
def split_annotations(original_json_path, files_moved, test_json_path, val_json_path):
    """Splits a COCO JSON annotation file into test and validation sets."""
    print(f"Splitting COCO annotations from: {original_json_path.name}")
    try:
        with original_json_path.open('r', encoding='utf-8') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading or parsing {original_json_path}: {e}")
        return

    images_map = {img['file_name']: img for img in data['images']}
    
    test_images = [images_map[fname] for fname in files_moved if fname in images_map]
    val_images = [img for img in data['images'] if img['file_name'] not in files_moved]
    
    test_image_ids = {img['id'] for img in test_images}
    
    test_annotations = [ann for ann in data['annotations'] if ann['image_id'] in test_image_ids]
    val_annotations = [ann for ann in data['annotations'] if ann['image_id'] not in test_image_ids]

    test_split = {'images': test_images, 'annotations': test_annotations, 'categories': data.get('categories', [])}
    val_split = {'images': val_images, 'annotations': val_annotations, 'categories': data.get('categories', [])}

    backup_path = original_json_path.with_suffix('.json.bak')
    shutil.copy(original_json_path, backup_path)
    print(f"Backed up original annotation to {backup_path}")

    with test_json_path.open('w', encoding='utf-8') as f: json.dump(test_split, f, indent=4)
    with val_json_path.open('w', encoding='utf-8') as f: json.dump(val_split, f, indent=4)
    print(f"Created new 'test.json' ({len(test_images)} images) and 'val.json' ({len(val_images)} images)")

def update_yaml_file():
    print("\n--- 3. Updating YAML Configuration File ---")
    yaml_path = Path('data/laboro_tomato/example_dataset.yaml')
    
    if not yaml_path.exists():
        print(f"Warning: Dataset YAML file not found at '{yaml_path}'. Cannot update.")
        return

    try:
        lines = yaml_path.read_text(encoding='utf-8').splitlines()
        
        preserved_lines = [
            line for line in lines 
            if not line.strip().startswith(('path:', 'train:', 'val:', 'test:'))
        ]

        # Correct paths relative to the YAML file's location
        new_path_lines = [
            "train: train/images",
            "val: val/images",
            "test: test/images"
        ]

        final_content = '\n'.join(preserved_lines + new_path_lines)
        yaml_path.write_text(final_content + '\n', encoding='utf-8')
        print("YAML file updated successfully.")
        
    except Exception as e:
        print(f"An error occurred while updating the YAML file: {e}")

def main():
    """Main function to run the full dataset setup process."""
    print("=======================================")
    print("  Starting Full Dataset Setup Process  ")
    
    if download_and_move_dataset():
        if split_dataset_files():
            update_yaml_file()

    print("\n=======================================")
    print("      Dataset Setup Process Finished     ")

if __name__ == '__main__':
    main()
