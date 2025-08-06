import os
import kagglehub
import shutil
import random
import json
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
        source_path_str = kagglehub.dataset_download("nexuswho/laboro-tomato")
        source_path = Path(source_path_str)
        print(f"Dataset downloaded to cache: {source_path}")

        dest_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Moving dataset files from cache to '{dest_dir}'...")
        for item in source_path.iterdir():
            dest_item_path = dest_dir / item.name
            print(f"  - Moving {item.name}")
            shutil.move(str(item), str(dest_item_path))
            
        print("Dataset moved successfully.")
        return True

    except Exception as e:
        print(f"\nAn error occurred during download or move: {e}")
        return False

def split_coco_annotations(original_json_path, files_moved, test_json_path, val_json_path):
    """Splits a COCO JSON annotation file into test and validation sets."""
    print(f"Splitting COCO annotations from: {original_json_path.name}")
    try:
        with open(original_json_path, 'r', encoding='utf-8') as f:
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

    with open(test_json_path, 'w', encoding='utf-8') as f: json.dump(test_split, f, indent=4)
    with open(val_json_path, 'w', encoding='utf-8') as f: json.dump(val_split, f, indent=4)
    print(f"Created new 'test.json' ({len(test_images)} images) and 'val.json' ({len(val_images)} images)")

def split_dataset_files():
    """Splits the validation set into new val and test sets (50/50 split)."""
    print("\n--- 2. Preparing and Splitting Dataset ---")
    
    base_dir = Path('data/laboro_tomato')
    val_img_dir = base_dir / 'val/images'
    test_img_dir = base_dir / 'test/images'

    if test_img_dir.exists() and any(test_img_dir.iterdir()):
        print("'test' directory already exists and is not empty. Skipping split.")
        return

    val_lbl_dir = base_dir / 'val/labels'
    test_lbl_dir = base_dir / 'test/labels'
    annotations_dir = base_dir / 'annotations'
    yaml_path = base_dir / 'example_dataset.yaml'
    
    test_img_dir.mkdir(parents=True, exist_ok=True)
    test_lbl_dir.mkdir(parents=True, exist_ok=True)

    val_images = [f.name for f in val_img_dir.iterdir() if f.is_file()]
    random.shuffle(val_images)
    files_to_move = val_images[:len(val_images) // 2]
    
    if not files_to_move:
        print("Not enough files in validation set to create a test split.")
        return

    print(f"Moving {len(files_to_move)} files from 'val' to 'test' directories.")
    for filename in files_to_move:
        base, _ = os.path.splitext(filename)
        shutil.move(str(val_img_dir / filename), str(test_img_dir / filename))
        if (val_lbl_dir / f"{base}.txt").exists():
            shutil.move(str(val_lbl_dir / f"{base}.txt"), str(test_lbl_dir / f"{base}.txt"))

    original_val_json = annotations_dir / 'test.json'
    if original_val_json.exists():
        split_coco_annotations(original_val_json, files_to_move, annotations_dir / 'test.json', annotations_dir / 'val.json')

    try:
        lines = yaml_path.read_text(encoding='utf-8').splitlines()
        if not any('test:' in line for line in lines):
            with open(yaml_path, 'a', encoding='utf-8') as f:
                f.write('\ntest: test/images\n')
            print(f"Updated '{yaml_path}' with test path.")
    except FileNotFoundError:
        print(f"Warning: Dataset YAML file not found at '{yaml_path}'")
    
    print("--- Dataset Preparation Complete ---")

def main():
    """Main function to run the full dataset setup process."""
    print("=======================================")
    print("  Starting Full Dataset Setup Process  ")
    print("=======================================")
    
    if download_and_move_dataset():
        split_dataset_files()

    print("\n=======================================")
    print("      Dataset Setup Process Finished     ")
    print("=======================================")

if __name__ == '__main__':
    main()