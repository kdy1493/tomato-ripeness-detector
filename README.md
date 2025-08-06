# Smart Farm - 토마토 탐지 및 상태 분류 모델

YOLOv10을 활용한 스마트팜 토마토 자동 탐지 및 성숙도 분류 시스템

## 프로젝트 개요

이 프로젝트는 농업 자동화를 위한 컴퓨터 비전 시스템으로, 토마토의 위치를 탐지하고 성숙도(green, half-ripened, fully-ripened)를 자동으로 분류합니다.

### 주요 기능
- **토마토 객체 탐지**: YOLOv10 기반 실시간 토마토 위치 인식
- **성숙도 분류**: 6가지 클래스 (방울토마토/대형토마토 × 녹색/반쯤 익음/완전히 익음)
- **고성능 모델**: mAP@0.5 81.6% 달성

## 프로젝트 구조

```
smart-farm/
├── .project-root          # 자동 경로 설정을 위한 마커 파일
├── data/
│   └── laboro_tomato/     # 데이터셋 루트
├── src/
│   ├── Yolo_train.py      # 모델 학습 스크립트
│   ├── Yolo_predict.py    # 최종 모델 테스트 및 결과 생성 스크립트
│   ├── fiftyone_viewer.py # FiftyOne을 사용한 결과 시각화 스크립트
│   ├── data_download.py   # 원본 데이터셋 다운로드
│   └── prepare_dataset.py # 데이터셋 분할 및 준비 스크립트
├── predict_results/       # 테스트 예측 결과 저장소
├── train_results/         # 학습 결과 저장소
├── pyproject.toml         # 프로젝트 의존성 및 설정
└── README.md
```

## 성능 지표

`yolov10n_baseline_train` 모델을 `test` 데이터셋으로 평가한 최종 성능입니다.

| 지표 | 값 |
| :--- | :--- |
| **mAP@0.5** | 81.6% |
| **mAP@0.5-0.95** | 68.8% |

### 클래스별 성능 (mAP@0.5)

- `b_green` (녹색 방울토마토): 91.1%
- `l_green` (녹색 대형토마토): 84.7%
- `l_fully_ripened` (익은 대형토마토): 82.8%
- `b_half_ripened` (반쯤 익은 방울토마토): 80.9%
- `l_half_ripened` (반쯤 익은 대형토마토): 76.2%
- `b_fully_ripened` (익은 방울토마토): 74.1%

## 시작하기

### 1. 환경 설정

```bash
# 저장소 클론
git clone <repository-url>
cd smart-farm

# Python 가상환경 생성 및 활성화
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 또는
.venv\Scripts\activate     # Windows

# 의존성 설치 (uv 사용 권장)
uv sync
```

### 2. 데이터셋 준비

```bash
# 데이터셋 다운로드
python src/data_download.py

# 2. 검증(val) 데이터를 검증/테스트(val/test)로 분할
python src/prepare_dataset.py
```

### 3. 모델 학습

`val` 데이터셋의 구성이 변경되었으므로, 모델을 재학습하여 새로운 `val` 데이터에 최적화된 모델을 생성합니다.

```bash
# 모델 재학습 실행
python src/Yolo_train.py
```
학습이 완료되면 `YOLO_train_results/yolov10n_baseline_train/` 폴더에 결과와 모델 가중치(`best.pt`)가 저장됩니다.

### 4. 최종 모델 평가

재학습된 모델을 사용하여, 학습에 전혀 사용되지 않은 `test` 데이터셋으로 최종 성능을 평가합니다.

```bash
# 테스트 데이터셋으로 예측 실행
# 결과는 YOLO_predict_results/yolov10n_baseline_predict/ 폴더에 저장됩니다.
python src/Yolo_predict.py
```
이 스크립트는 전체 예측 결과가 담긴 `predictions.json` 파일과 샘플 예측 이미지(`test_sample_prediction.jpg`)를 생성합니다.

### 5. 결과 시각화 및 분석

`fiftyone`을 사용하여 정답 데이터와 모델의 예측 결과를 한눈에 비교하고 분석합니다.

```bash
# FiftyOne 시각화 앱 실행
python src/fiftyone_viewer.py
```
웹 브라우저가 열리며, 각 테스트 이미지에 대한 정답(ground_truth)과 예측(predictions) 바운딩 박스를 함께 확인할 수 있습니다.

## 기술 스택

- **AI/ML**: Ultralytics YOLOv10, PyTorch
- **데이터 분석/시각화**: FiftyOne
- **경로 관리**: autorootcwd
- **데이터 처리**: OpenCV, PyYAML
- **언어**: Python 3.11+
- **패키지 관리**: uv
