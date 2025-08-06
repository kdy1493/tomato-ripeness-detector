import os
import shutil
import random
import json
from pathlib import Path

def split_coco_annotations(original_json_path, files_moved_to_test, test_json_path, val_json_path):
    """
    Splits a COCO-style JSON annotation file into test and validation sets based on the list of moved files.
    """
    print(f"\n--- Splitting COCO Annotation File: {original_json_path.name} ---")

    try:
        with open(original_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading or parsing {original_json_path}: {e}")
        return

    # Create mappings for quick lookups
    images_map = {img['file_name']: img for img in data['images']}
    
    # Separate images into test and val sets based on moved files
    test_images = [images_map[fname] for fname in files_moved_to_test if fname in images_map]
    val_images = [img for img in data['images'] if img['file_name'] not in files_moved_to_test]
    
    test_image_ids = {img['id'] for img in test_images}
    val_image_ids = {img['id'] for img in val_images}
    
    # Separate annotations based on image IDs
    test_annotations = [ann for ann in data['annotations'] if ann['image_id'] in test_image_ids]
    val_annotations = [ann for ann in data['annotations'] if ann['image_id'] in val_image_ids]

    # Create new JSON structures
    categories = data.get('categories', [])
    test_split = {'images': test_images, 'annotations': test_annotations, 'categories': categories}
    val_split = {'images': val_images, 'annotations': val_annotations, 'categories': categories}

    # Backup the original annotation file before overwriting
    backup_path = original_json_path.with_suffix('.json.bak')
    print(f"Backing up original annotation file to {backup_path}")
    shutil.copy(original_json_path, backup_path)

    # Write new JSON files
    try:
        with open(test_json_path, 'w', encoding='utf-8') as f:
            json.dump(test_split, f, indent=4)
        print(f"Successfully created new test annotation file: {test_json_path} ({len(test_images)} images)")
        
        with open(val_json_path, 'w', encoding='utf-8') as f:
            json.dump(val_split, f, indent=4)
        print(f"Successfully created new val annotation file: {val_json_path} ({len(val_images)} images)")

    except IOError as e:
        print(f"Error writing new annotation files: {e}")

def prepare_dataset():
    """
    Splits the validation dataset into a new validation and test dataset.
    50% of the validation files will be moved to the test set.
    This handles images, YOLO labels, and COCO JSON annotations.
    """
    print("--- Starting Dataset Preparation ---")
    
    # --- 1. 경로 설정 (pathlib 사용) ---
    base_dir = Path('data/laboro_tomato')
    val_img_dir = base_dir / 'val/images'
    val_lbl_dir = base_dir / 'val/labels'
    test_img_dir = base_dir / 'test/images'
    test_lbl_dir = base_dir / 'test/labels'
    annotations_dir = base_dir / 'annotations'
    yaml_path = base_dir / 'example_dataset.yaml'
    
    # The original JSON for the validation set is misnamed 'test.json'
    original_val_json_path = annotations_dir / 'test.json'

    # --- 2. test 폴더가 이미 존재하는지 확인 ---
    if test_img_dir.exists() and any(test_img_dir.iterdir()):
        print(f"'{test_img_dir}' already exists and is not empty. Skipping preparation.")
        return

    print("Test directory not found or is empty. Proceeding to create test split.")
    
    # --- 3. 새 폴더 생성 ---
    test_img_dir.mkdir(parents=True, exist_ok=True)
    test_lbl_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created directory: {test_img_dir}")
    print(f"Created directory: {test_lbl_dir}")

    # --- 4. 파일 목록 가져오기 및 분리 ---
    if not val_img_dir.exists() or not any(val_img_dir.iterdir()):
        print(f"Error: Validation directory '{val_img_dir}' is empty or does not exist.")
        return
        
    val_images = [f.name for f in val_img_dir.iterdir()]
    random.shuffle(val_images)
    num_to_move = len(val_images) // 2
    files_to_move = val_images[:num_to_move]
    
    if num_to_move == 0:
        print("Not enough files in validation set to create a test split.")
        return
        
    print(f"Total validation images: {len(val_images)}. Moving {len(files_to_move)} files to test set.")

    # --- 5. 파일 이동 (Images and YOLO .txt labels) ---
    moved_count = 0
    for filename in files_to_move:
        base, ext = os.path.splitext(filename)
        
        # 이미지 파일 이동
        src_img_path = val_img_dir / filename
        dst_img_path = test_img_dir / filename
        if src_img_path.exists():
            shutil.move(str(src_img_path), str(dst_img_path))
            moved_count += 1
        
        # 라벨 파일(.txt) 이동
        src_lbl_path_txt = val_lbl_dir / f"{base}.txt"
        dst_lbl_path_txt = test_lbl_dir / f"{base}.txt"
        if src_lbl_path_txt.exists():
            shutil.move(str(src_lbl_path_txt), str(dst_lbl_path_txt))

    print(f"Moved {moved_count} image and corresponding label files to test directories.")

    # --- 6. COCO JSON 어노테이션 분리 ---
    new_test_json_path = annotations_dir / 'test.json'
    new_val_json_path = annotations_dir / 'val.json'
    
    if original_val_json_path.exists():
        split_coco_annotations(
            original_json_path=original_val_json_path,
            files_moved_to_test=files_to_move,
            test_json_path=new_test_json_path,
            val_json_path=new_val_json_path
        )
    else:
        print(f"Warning: Annotation file {original_val_json_path} not found. Skipping annotation split.")

    # --- 7. YAML 파일 업데이트 ---
    try:
        lines = yaml_path.read_text(encoding='utf-8').splitlines()
        if not any('test:' in line for line in lines):
            with open(yaml_path, 'a', encoding='utf-8') as f:
                f.write('\ntest: test/images\n')
            print(f"Updated '{yaml_path}' with test path.")
        else:
            print(f"'{yaml_path}' already has a test path.")
            
    except FileNotFoundError:
        print(f"Error: Dataset YAML file not found at '{yaml_path}'")

    print("--- Dataset Preparation Complete ---")


if __name__ == '__main__':
    prepare_dataset()
