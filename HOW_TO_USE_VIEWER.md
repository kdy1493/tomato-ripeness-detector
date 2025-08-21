# FiftyOne 데이터 시각화 스크립트 사용법 (`fiftyone_viewer_with_config.py`)

이 문서는 `fiftyone_viewer_with_config.py` 스크립트를 사용하여 자신만의 객체 탐지(Object Detection) 데이터셋을 시각화하는 방법을 안내합니다.

## 1. 스크립트 소개

이 스크립트는 이미지와 라벨링 정보(어노테이션)를 FiftyOne 툴을 통해 시각화해주는 도구입니다. 모델의 예측 결과를 함께 로드하여 정답과 비교 분석하는 것도 가능합니다. 모든 실행 옵션은 `config.yaml` 파일을 통해 관리되므로, 터미널 명령어를 간결하게 유지할 수 있는 장점이 있습니다.

**주요 기능:**
- Ground Truth (정답) 데이터셋 시각화
- 모델 예측(Prediction) 결과와 정답 데이터 비교
- YAML 설정 파일을 통한 간편한 실행
- (선택) CVAT와의 연동을 통한 데이터 라벨링 및 결과 불러오기

---

## 2. 사전 준비

스크립트를 사용하기 전에, 시각화할 데이터가 특정 형식에 맞게 준비되어 있어야 합니다.

### 필수 구성 요소

새로운 데이터를 시각화하기 위해서는 아래 **3가지 요소**가 반드시 필요합니다.

#### 1) 이미지 파일 (`.jpg`, `.png` 등)
- 원본 이미지 파일들입니다.
- 보통 `train`, `val`, `test`와 같이 용도에 따라 폴더를 구분하여 관리하는 것을 권장합니다.
- **예시 폴더 구조:**
  ```
  my_dataset/
  ├── train/
  │   ├── 00001.jpg
  │   └── 00002.jpg
  └── val/
      ├── 00003.jpg
      └── 00004.jpg
  ```

#### 2) COCO 형식 어노테이션 파일 (`.json`)
- 각 이미지에 어떤 객체가 어디 있는지에 대한 정답 정보가 담긴 파일입니다.
- 데이터는 **MS COCO 데이터셋 형식**을 따라야 합니다.
- 각 스플릿(`train`, `val` 등)에 대해 별도의 JSON 파일을 준비해야 합니다. (예: `train.json`, `val.json`)

- **`train.json` 파일의 핵심 구조:**
  ```json
  {
    "images": [
      { "id": 1, "file_name": "00001.jpg", "width": 640, "height": 480 },
      ...
    ],
    "annotations": [
      {
        "id": 1,
        "image_id": 1,
        "category_id": 2,
        "bbox": [100, 150, 50, 75]
      },
      ...
    ],
    "categories": [
      { "id": 1, "name": "tomato" },
      { "id": 2, "name": "leaf" },
      ...
    ]
  }
  ```
  - `images`: 이미지 파일 목록과 고유 ID, 크기 정보.
  - `annotations`: 바운딩 박스 정보. `bbox`는 `[x, y, width, height]` 형식입니다.
  - `categories`: 클래스(객체 종류)의 이름과 ID. **ID는 1부터 시작하는 것을 권장합니다.**

#### 3) 데이터셋 설정 파일 (`.yaml`)
- 스크립트에게 데이터셋의 구조를 알려주는 역할을 합니다.
- **`my_dataset.yaml` 파일 예시:**
  ```yaml
  # 각 스플릿의 이미지 폴더 경로 (이 YAML 파일 기준 상대 경로)
  train: my_dataset/train/
  val: my_dataset/val/

  # 클래스 이름 목록 (COCO JSON의 category_id 순서와 일치해야 함)
  # category_id: 1 -> 'tomato', category_id: 2 -> 'leaf'
  names:
    - tomato
    - leaf
  ```
  - **중요:** `names` 목록의 순서는 `categories`의 `id` 순서와 반드시 일치해야 합니다. (0번째 이름이 `id: 1`, 1번째 이름이 `id: 2`에 해당)

---

## 3. 실행 방법

#### 1단계: `config.yaml` 파일 수정
프로젝트 루트에 있는 `config.yaml` 파일을 열고, 방금 준비한 자신만의 데이터 경로를 입력합니다.

- **`config.yaml` 수정 예시:**
  ```yaml
  # --- 필수 경로 설정 ---
  # Ground Truth COCO JSON 파일 경로
  gt: "path/to/your/train.json"

  # 데이터셋의 YAML 설정 파일 경로
  dataset_yaml: "path/to/your/my_dataset.yaml"

  # (선택) 모델 예측 결과 JSON 파일 경로
  # predictions: "path/to/your/predictions.json"

  # --- 기본 실행 옵션 ---
  # gt 파일명에 train/val/test가 있으면 자동 유추되므로 생략 가능
  # split: "train"
  ```

#### 2단계: 스크립트 실행
터미널에서 아래 명령어를 실행합니다.

```bash
python src/fiftyone_viewer_with_config.py --config config.yaml
```

스크립트가 실행되면, 잠시 후 웹 브라우저에 자동으로 FiftyOne 뷰어 창이 열리며 시각화된 데이터를 확인할 수 있습니다.

---

## 4. 고급 기능: CVAT 연동

(이 기능을 사용하려면 `fiftyone-cvat` 플러그인 설치가 필요합니다.)

FiftyOne에서 확인한 데이터를 CVAT로 보내 라벨링을 수정하거나, CVAT에서 작업한 내용을 다시 불러올 수 있습니다.

#### CVAT 연동 설정
- `config.yaml` 파일의 CVAT 관련 옵션을 활성화합니다.
  ```yaml
  # --- CVAT 연동 옵션 ---
  to_cvat: true  # FiftyOne -> CVAT로 데이터 업로드
  from_cvat: false # CVAT -> FiftyOne으로 데이터 다운로드

  anno_key: "my_first_labeling_job" # 작업을 식별할 고유한 이름
  cvat_url: "https://app.cvat.ai" # 사용 중인 CVAT 주소
  ```

#### CVAT 인증 (로그인)
CVAT 서버에 로그인하기 위한 정보는 **환경 변수**로 설정하는 것이 가장 안전하고 편리합니다.

- **일반 ID/PW 사용 시:**
  ```powershell
  $env:FIFTYONE_CVAT_USERNAME = "your_cvat_username"
  $env:FIFTYONE_CVAT_PASSWORD = "your_cvat_password"
  ```

- **구글 로그인(SSO) 등 API 키 사용 시:**
  CVAT 웹사이트의 프로필 메뉴에서 API 키를 발급받아, 비밀번호 대신 사용해야 합니다.
  ```powershell
  $env:FIFTYONE_CVAT_USERNAME = "your_cvat_username"
  $env:FIFTYONE_CVAT_PASSWORD = "your_cvat_api_key_paste_here"
  ```
환경 변수를 설정한 후 스크립트를 실행하면, 별도의 로그인 프롬프트 없이 자동으로 인증을 시도합니다.
