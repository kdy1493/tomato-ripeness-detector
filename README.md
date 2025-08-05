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
├── src/
│   ├── train.py           # 모델 학습 스크립트
│   ├── predict.py         # 추론 실행 스크립트
│   └── data_download.py   # 데이터셋 다운로드
├── models/
│   ├── yolov10n.pt       # YOLOv10 기본 모델
│   └── yolo11n.pt        # YOLO11 기본 모델
├── train_results/         # 학습 결과 저장소
├── pyproject.toml        # 프로젝트 의존성 설정
└── README.md
```

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
```

### 3. 모델 학습

```bash
# 토마토 분류 모델 학습 (100 에포크)
python src/train.py
```

### 4. 추론 실행

```bash
# 학습된 모델로 토마토 탐지
python src/predict.py
```

## 성능 지표

최신 `yolov10n_baseline` 모델의 검증 결과입니다.

| 지표 | 값 |
|------|-----|
| **mAP@0.5** | 81.6% |
| **mAP@0.5-0.95** | 68.8% |

### 클래스별 성능 (mAP@0.5)
- `b_green` (녹색 방울토마토): 91.1%
- `l_green` (녹색 대형토마토): 84.7%
- `l_fully_ripened` (익은 대형토마토): 82.8%
- `b_half_ripened` (반쯤 익은 방울토마토): 80.9%
- `l_half_ripened` (반쯤 익은 대형토마토): 76.2%
- `b_fully_ripened` (익은 방울토마토): 74.1%

## 사용법

### 기본 추론
```python
from ultralytics import YOLO

# 모델 로드
model = YOLO('train_results/yolov10n_baseline/weights/best.pt')

# 이미지 예측
results = model('path/to/tomato_image.jpg')

# 결과 시각화
results[0].show()
```

### 실시간 추론
```python
# 웹캠 실시간 탐지
results = model(source=0, show=True)  # 0번 카메라 사용
```

## 학습 결과 분석

학습 완료 후 `train_results/yolov10n_baseline/` 폴더에서 다음 파일들을 확인할 수 있습니다:

- `results.png`: 학습 과정 종합 그래프
- `confusion_matrix.png`: 클래스별 예측 정확도 행렬
- `BoxF1_curve.png`: F1-점수 커브 (최적 성능 지점 확인)
- `val_batch*_pred.jpg`: 검증 데이터 예측 결과 시각화

## 기술 스택

- **딥러닝 프레임워크**: Ultralytics YOLO
- **모델 아키텍처**: YOLOv10n
- **이미지 처리**: OpenCV
- **데이터**: Laboro Tomato Dataset
- **언어**: Python 3.11+

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 기여하기

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

