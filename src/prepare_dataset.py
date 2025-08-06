import os
import shutil
import random

def prepare_dataset():
    """
    Splits the validation dataset into a new test dataset if it doesn't already exist.
    50% of the validation files will be moved to the test set.
    """
    print("--- Starting Dataset Preparation ---")
    
    # --- 1. 경로 설정 ---
    base_dir = 'data/laboro_tomato'
    val_img_dir = os.path.join(base_dir, 'val/images')
    val_lbl_dir = os.path.join(base_dir, 'val/labels')
    test_img_dir = os.path.join(base_dir, 'test/images')
    test_lbl_dir = os.path.join(base_dir, 'test/labels')
    yaml_path = os.path.join(base_dir, 'example_dataset.yaml')

    # --- 2. test 폴더가 이미 존재하는지 확인 ---
    if os.path.exists(test_img_dir) and os.listdir(test_img_dir):
        print("'test' directory already exists and is not empty. Skipping preparation.")
        return

    print("Test directory not found or is empty. Proceeding to create test split.")
    
    # --- 3. 새 폴더 생성 ---
    os.makedirs(test_img_dir, exist_ok=True)
    os.makedirs(test_lbl_dir, exist_ok=True)
    print(f"Created directory: {test_img_dir}")
    print(f"Created directory: {test_lbl_dir}")

    # --- 4. 파일 목록 가져오기 및 분리 ---
    if not os.path.exists(val_img_dir) or not os.listdir(val_img_dir):
        print(f"Error: Validation directory '{val_img_dir}' is empty or does not exist.")
        print("Please ensure you have downloaded and unzipped the dataset correctly.")
        return
        
    val_images = os.listdir(val_img_dir)
    random.shuffle(val_images)
    num_to_move = len(val_images) // 2
    files_to_move = val_images[:num_to_move]
    
    if num_to_move == 0:
        print("Not enough files in validation set to create a test split.")
        return
        
    print(f"Total validation images: {len(val_images)}. Moving {len(files_to_move)} files to test set.")

    # --- 5. 파일 이동 ---
    for filename in files_to_move:
        base, ext = os.path.splitext(filename)
        
        # 이미지 파일 이동
        src_img_path = os.path.join(val_img_dir, filename)
        dst_img_path = os.path.join(test_img_dir, filename)
        shutil.move(src_img_path, dst_img_path)
        
        # 라벨 파일 이동 (YOLO .txt)
        src_lbl_path_txt = os.path.join(val_lbl_dir, base + '.txt')
        dst_lbl_path_txt = os.path.join(test_lbl_dir, base + '.txt')
        if os.path.exists(src_lbl_path_txt):
            shutil.move(src_lbl_path_txt, dst_lbl_path_txt)

    print(f"Moved {len(files_to_move)} image and label files to test directories.")

    # --- 6. YAML 파일 업데이트 ---
    try:
        with open(yaml_path, 'r') as f:
            lines = f.readlines()

        has_test_path = any('test:' in line for line in lines)

        if not has_test_path:
            with open(yaml_path, 'a') as f:
                f.write('\ntest: test/images\n')
            print(f"Updated '{yaml_path}' with test path.")
        else:
            print(f"'{yaml_path}' already has a test path.")
            
    except FileNotFoundError:
        print(f"Error: Dataset YAML file not found at '{yaml_path}'")

    print("--- Dataset Preparation Complete ---")


if __name__ == '__main__':
    prepare_dataset()
