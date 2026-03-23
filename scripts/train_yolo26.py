"""
YOLO26 학습 스크립트
====================
Ultralytics YOLO26 모델을 커스텀 데이터셋으로 학습시키는 코드입니다.

사용법:
    # YAML config 사용
    python train_yolo26.py -c config/train_yolo26.yaml
    
    # 모드 직접 지정
    python train_yolo26.py -c config/train_yolo26.yaml --mode tune

YOLO26 주요 특징:
- NMS-free 추론 (End-to-End)
- Distribution Focal Loss (DFL) 제거
- ProgLoss + STAL (소형 객체 감지 향상)
- MuSGD 옵티마이저
- 최대 43% 빠른 CPU 추론
"""

import os
import yaml
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List

from ultralytics import YOLO


class YOLO26Trainer:
    """YOLO26 모델 학습을 위한 클래스"""
    
    # 사용 가능한 YOLO26 모델 크기
    AVAILABLE_MODELS = {
        'nano': 'yolo26n.pt',
        'small': 'yolo26s.pt',
        'medium': 'yolo26m.pt',
        'large': 'yolo26l.pt',
        'xlarge': 'yolo26x.pt',
    }
    
    # 지원하는 태스크
    AVAILABLE_TASKS = {
        'detect': '객체 감지',
        'segment': '인스턴스 분할',
        'pose': '포즈 추정',
        'obb': 'Oriented Bounding Box',
        'classify': '이미지 분류',
    }
    
    # 튜닝 가능한 하이퍼파라미터 기본 탐색 공간
    DEFAULT_TUNE_SPACE = {
        'lr0': (1e-5, 1e-1),           # 초기 학습률
        'lrf': (0.01, 1.0),            # 최종 학습률 비율
        'momentum': (0.6, 0.98),       # 모멘텀
        'weight_decay': (0.0, 0.001),  # 가중치 감소
        'warmup_epochs': (0.0, 5.0),   # 워밍업 에폭
        'warmup_momentum': (0.0, 0.95), # 워밍업 모멘텀
        'box': (0.02, 0.2),            # 박스 손실 가중치
        'cls': (0.2, 4.0),             # 분류 손실 가중치
        'hsv_h': (0.0, 0.1),           # HSV 색상
        'hsv_s': (0.0, 0.9),           # HSV 채도
        'hsv_v': (0.0, 0.9),           # HSV 명도
        'degrees': (0.0, 45.0),        # 회전 각도
        'translate': (0.0, 0.9),       # 이동
        'scale': (0.0, 0.9),           # 스케일
        'shear': (0.0, 10.0),          # 전단
        'perspective': (0.0, 0.001),   # 원근
        'flipud': (0.0, 1.0),          # 상하 반전
        'fliplr': (0.0, 1.0),          # 좌우 반전
        'mosaic': (0.0, 1.0),          # 모자이크
        'mixup': (0.0, 1.0),           # MixUp
        'copy_paste': (0.0, 1.0),      # Copy-paste
    }
    
    def __init__(self, 
                 model_size: str = 'nano',
                 task: str = 'detect',
                 pretrained: bool = True):
        """
        Args:
            model_size: 모델 크기 ('nano', 'small', 'medium', 'large', 'xlarge')
            task: 태스크 타입 ('detect', 'segment', 'pose', 'obb', 'classify')
            pretrained: 사전학습 가중치 사용 여부
        """
        self.model_size = model_size
        self.task = task
        self.pretrained = pretrained
        self.model = None
        self.results = None
        
        # 모델 이름 설정
        if task == 'detect':
            self.model_name = self.AVAILABLE_MODELS.get(model_size, 'yolo26n.pt')
        elif task == 'segment':
            self.model_name = f"yolo26{model_size[0]}-seg.pt"
        elif task == 'pose':
            self.model_name = f"yolo26{model_size[0]}-pose.pt"
        elif task == 'obb':
            self.model_name = f"yolo26{model_size[0]}-obb.pt"
        elif task == 'classify':
            self.model_name = f"yolo26{model_size[0]}-cls.pt"
        
        print(f"YOLO26 트레이너 초기화")
        print(f"  - 모델: {self.model_name}")
        print(f"  - 태스크: {self.AVAILABLE_TASKS.get(task, task)}")
        print(f"  - 사전학습: {'사용' if pretrained else '미사용'}")
    
    def load_model(self):
        """YOLO26 모델 로드"""
        print(f"\n모델 로딩 중: {self.model_name}")
        
        if self.pretrained:
            self.model = YOLO(self.model_name)
        else:
            config_name = self.model_name.replace('.pt', '.yaml')
            self.model = YOLO(config_name)
        
        print("모델 로드 완료!")
        return self.model
    
    def train(self,
              data_yaml: str,
              epochs: int = 100,
              imgsz: int = 640,
              batch: int = 16,
              device: str = None,
              workers: int = 8,
              patience: int = 50,
              save_period: int = 10,
              project: str = 'runs/train',
              name: str = None,
              exist_ok: bool = False,
              resume: bool = False,
              optimizer: str = 'auto',
              lr0: float = 0.01,
              lrf: float = 0.01,
              momentum: float = 0.937,
              weight_decay: float = 0.0005,
              warmup_epochs: float = 3.0,
              warmup_momentum: float = 0.8,
              warmup_bias_lr: float = 0.1,
              box: float = 7.5,
              cls: float = 0.5,
              dfl: float = 1.5,
              hsv_h: float = 0.015,
              hsv_s: float = 0.7,
              hsv_v: float = 0.4,
              degrees: float = 0.0,
              translate: float = 0.1,
              scale: float = 0.5,
              shear: float = 0.0,
              perspective: float = 0.0,
              flipud: float = 0.0,
              fliplr: float = 0.5,
              mosaic: float = 1.0,
              mixup: float = 0.0,
              copy_paste: float = 0.0,
              close_mosaic: int = 10,
              amp: bool = True,
              val: bool = True,
              plots: bool = True,
              cache: bool = False,
              verbose: bool = True,
              seed: int = 0,
              **kwargs):
        """YOLO26 모델 학습"""
        if self.model is None:
            self.load_model()
        
        # 디바이스 자동 감지
        if device is None:
            device = '0' if torch.cuda.is_available() else 'cpu'
        
        # 실험 이름 자동 생성
        if name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = f"yolo26_{self.model_size}_{timestamp}"
        
        print(f"\n{'='*60}")
        print(f"YOLO26 학습 시작")
        print(f"{'='*60}")
        print(f"  - 데이터셋: {data_yaml}")
        print(f"  - 에폭: {epochs}")
        print(f"  - 이미지 크기: {imgsz}")
        print(f"  - 배치 크기: {batch}")
        print(f"  - 디바이스: {device}")
        print(f"  - 결과 저장: {project}/{name}")
        print(f"{'='*60}\n")
        
        # 학습 실행
        self.results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            workers=workers,
            patience=patience,
            save_period=save_period,
            project=project,
            name=name,
            exist_ok=exist_ok,
            resume=resume,
            optimizer=optimizer,
            lr0=lr0,
            lrf=lrf,
            momentum=momentum,
            weight_decay=weight_decay,
            warmup_epochs=warmup_epochs,
            warmup_momentum=warmup_momentum,
            warmup_bias_lr=warmup_bias_lr,
            box=box,
            cls=cls,
            dfl=dfl,
            hsv_h=hsv_h,
            hsv_s=hsv_s,
            hsv_v=hsv_v,
            degrees=degrees,
            translate=translate,
            scale=scale,
            shear=shear,
            perspective=perspective,
            flipud=flipud,
            fliplr=fliplr,
            mosaic=mosaic,
            mixup=mixup,
            copy_paste=copy_paste,
            close_mosaic=close_mosaic,
            amp=amp,
            val=val,
            plots=plots,
            cache=cache,
            verbose=verbose,
            seed=seed,
            **kwargs
        )
        
        print(f"\n학습 완료!")
        print(f"결과 저장 위치: {project}/{name}")
        
        return self.results
    
    def tune(self,
             data_yaml: str,
             epochs: int = 30,
             iterations: int = 100,
             imgsz: int = 640,
             batch: int = 16,
             device: str = None,
             workers: int = 8,
             optimizer: str = 'AdamW',
             project: str = 'runs/tune',
             name: str = None,
             exist_ok: bool = True,
             space: Dict[str, Tuple[float, float]] = None,
             use_ray: bool = False,
             gpu_per_trial: int = None,
             max_samples: int = 10,
             grace_period: int = 10,
             plots: bool = True,
             save: bool = False,
             val: bool = False,
             verbose: bool = True,
             seed: int = 0,
             **kwargs) -> Dict:
        """
        하이퍼파라미터 자동 튜닝
        
        Args:
            data_yaml: 데이터셋 설정 YAML 파일 경로
            epochs: 각 튜닝 반복당 에폭 수
            iterations: 튜닝 반복 횟수 (Genetic Algorithm)
            imgsz: 입력 이미지 크기
            batch: 배치 크기
            device: 디바이스
            workers: 데이터 로더 워커 수
            optimizer: 옵티마이저 ('SGD', 'Adam', 'AdamW')
            project: 결과 저장 프로젝트 폴더
            name: 실험 이름
            exist_ok: 기존 폴더 덮어쓰기
            space: 하이퍼파라미터 탐색 공간 딕셔너리
                   예: {'lr0': (1e-5, 1e-1), 'momentum': (0.6, 0.98)}
            use_ray: Ray Tune 사용 여부 (고급 튜닝)
            gpu_per_trial: Ray Tune 사용 시 트라이얼당 GPU 수
            max_samples: Ray Tune 사용 시 최대 샘플 수
            grace_period: Ray Tune ASHA 스케줄러 grace period
            plots: 플롯 생성 여부
            save: 각 반복마다 체크포인트 저장
            val: 각 반복마다 검증 수행
            verbose: 상세 출력
            seed: 랜덤 시드
            
        Returns:
            튜닝 결과 딕셔너리
        """
        if self.model is None:
            self.load_model()
        
        # 디바이스 자동 감지
        if device is None:
            device = '0' if torch.cuda.is_available() else 'cpu'
        
        # 실험 이름 자동 생성
        if name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = f"tune_{self.model_size}_{timestamp}"
        
        # 탐색 공간 설정
        if space is None:
            # 주요 하이퍼파라미터만 기본 탐색
            space = {
                'lr0': (1e-5, 1e-1),
                'lrf': (0.01, 1.0),
                'momentum': (0.6, 0.98),
                'weight_decay': (0.0, 0.001),
                'box': (0.02, 0.2),
                'cls': (0.2, 4.0),
            }
        
        print(f"\n{'='*60}")
        print(f"YOLO26 하이퍼파라미터 튜닝")
        print(f"{'='*60}")
        print(f"  - 데이터셋: {data_yaml}")
        print(f"  - 반복 횟수: {iterations}")
        print(f"  - 반복당 에폭: {epochs}")
        print(f"  - 이미지 크기: {imgsz}")
        print(f"  - 배치 크기: {batch}")
        print(f"  - 디바이스: {device}")
        print(f"  - Ray Tune: {'사용' if use_ray else '미사용'}")
        print(f"  - 결과 저장: {project}/{name}")
        print(f"\n[탐색 공간]")
        for param, (low, high) in space.items():
            print(f"  - {param}: ({low}, {high})")
        print(f"{'='*60}\n")
        
        # Ray Tune 사용
        if use_ray:
            try:
                import ray
                from ray import tune as ray_tune
                print("Ray Tune을 사용한 고급 튜닝 시작...")
                
                # Ray Tune 형식으로 탐색 공간 변환
                ray_space = {}
                for param, (low, high) in space.items():
                    ray_space[param] = ray_tune.uniform(low, high)
                
                results = self.model.tune(
                    data=data_yaml,
                    epochs=epochs,
                    iterations=iterations,
                    imgsz=imgsz,
                    batch=batch,
                    device=device,
                    workers=workers,
                    optimizer=optimizer,
                    project=project,
                    name=name,
                    exist_ok=exist_ok,
                    space=ray_space,
                    use_ray=True,
                    gpu_per_trial=gpu_per_trial,
                    max_samples=max_samples,
                    grace_period=grace_period,
                    seed=seed,
                    **kwargs
                )
            except ImportError:
                print("Ray Tune이 설치되지 않았습니다. 기본 튜닝으로 전환합니다.")
                print("Ray Tune 설치: pip install 'ray[tune]'")
                use_ray = False
        
        # Genetic Algorithm 튜닝 (기본)
        if not use_ray:
            print("Genetic Algorithm을 사용한 튜닝 시작...")
            results = self.model.tune(
                data=data_yaml,
                epochs=epochs,
                iterations=iterations,
                imgsz=imgsz,
                batch=batch,
                device=device,
                workers=workers,
                optimizer=optimizer,
                project=project,
                name=name,
                exist_ok=exist_ok,
                space=space,
                plots=plots,
                save=save,
                val=val,
                seed=seed,
                **kwargs
            )
        
        print(f"\n{'='*60}")
        print(f"튜닝 완료!")
        print(f"{'='*60}")
        print(f"결과 저장 위치: {project}/{name}")
        print(f"\n최적 하이퍼파라미터는 {project}/{name}/best_hyperparameters.yaml 에서 확인하세요.")
        print(f"{'='*60}")
        
        return results
    
    def validate(self, data_yaml: str = None, **kwargs):
        """모델 검증"""
        if self.model is None:
            self.load_model()
        
        results = self.model.val(data=data_yaml, **kwargs)
        return results
    
    def predict(self, source, **kwargs):
        """예측 수행"""
        if self.model is None:
            self.load_model()
        
        results = self.model.predict(source=source, **kwargs)
        return results
    
    def export(self, format: str = 'onnx', **kwargs):
        """모델 내보내기"""
        if self.model is None:
            self.load_model()
        
        print(f"\n모델 내보내기: {format} 포맷")
        path = self.model.export(format=format, **kwargs)
        print(f"내보내기 완료: {path}")
        return path


def create_dataset_yaml(
    train_path: str,
    val_path: str,
    test_path: str = None,
    class_names: list = None,
    output_path: str = 'dataset.yaml'
):
    """데이터셋 YAML 설정 파일 생성"""
    if class_names is None:
        class_names = ['class_0']
    
    data = {
        'path': str(Path(train_path).parent.absolute()),
        'train': train_path,
        'val': val_path,
        'nc': len(class_names),
        'names': class_names,
    }
    
    if test_path:
        data['test'] = test_path
    
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    
    print(f"데이터셋 설정 파일 생성: {output_path}")
    return output_path


def load_config(config_path: str) -> dict:
    """YAML 설정 파일 로드"""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_path}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def parse_tune_space(space_config: Dict) -> Dict[str, Tuple[float, float]]:
    """Config의 tune space를 튜플 형식으로 변환"""
    parsed = {}
    for key, value in space_config.items():
        if isinstance(value, list) and len(value) == 2:
            parsed[key] = tuple(value)
        elif isinstance(value, tuple):
            parsed[key] = value
        else:
            print(f"Warning: '{key}' 탐색 공간 형식이 올바르지 않습니다. 건너뜁니다.")
    return parsed


# ============================================================================
# 메인 실행
# ============================================================================
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLO26 학습 스크립트')
    parser.add_argument('-c', '--config', type=str, default='config/train_yolo26.yaml',
                        help='학습 설정 YAML 파일 경로')
    parser.add_argument('--mode', type=str, default=None,
                        choices=['train', 'tune', 'validate', 'predict', 'export'],
                        help='실행 모드 (config 파일 설정보다 우선)')
    
    args = parser.parse_args()
    
    print("""
╔═══════════════════════════════════════════════════════════════╗
║                     YOLO26 학습 스크립트                        ║
║                                                               ║
║  모드:                                                         ║
║  • train    : 모델 학습                                        ║
║  • tune     : 하이퍼파라미터 자동 튜닝                           ║
║  • validate : 모델 검증                                        ║
║  • predict  : 추론                                             ║
║  • export   : 모델 내보내기                                     ║
╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # 설정 파일 로드
    try:
        config = load_config(args.config)
        print(f"설정 파일 로드: {args.config}")
    except FileNotFoundError as e:
        print(f"오류: {e}")
        exit(1)
    
    # 모델 설정
    model_config = config.get('model', {})
    model_size = model_config.get('size', 'nano')
    task = model_config.get('task', 'detect')
    pretrained = model_config.get('pretrained', True)
    
    # 트레이너 생성
    trainer = YOLO26Trainer(
        model_size=model_size,
        task=task,
        pretrained=pretrained
    )
    trainer.load_model()
    
    # 실행 모드 결정 (CLI > config)
    mode = args.mode or config.get('mode', 'train')
    
    # =========================================================================
    # TRAIN 모드
    # =========================================================================
    if mode == 'train':
        train_config = config.get('train', {})
        data_yaml = train_config.get('data', 'data/merged_dataset/dataset.yaml')
        
        print(f"\n학습 시작: {data_yaml}")
        
        trainer.train(
            data_yaml=data_yaml,
            epochs=train_config.get('epochs', 100),
            imgsz=train_config.get('imgsz', 640),
            batch=train_config.get('batch', -1),
            device=train_config.get('device', None),
            workers=train_config.get('workers', 8),
            patience=train_config.get('patience', 50),
            save_period=train_config.get('save_period', 10),
            project=train_config.get('project', 'runs/train'),
            name=train_config.get('name', None),
            exist_ok=train_config.get('exist_ok', True),
            resume=train_config.get('resume', False),
            optimizer=train_config.get('optimizer', 'auto'),
            lr0=train_config.get('lr0', 0.01),
            lrf=train_config.get('lrf', 0.01),
            momentum=train_config.get('momentum', 0.937),
            weight_decay=train_config.get('weight_decay', 0.0005),
            warmup_epochs=train_config.get('warmup_epochs', 3.0),
            hsv_h=train_config.get('hsv_h', 0.015),
            hsv_s=train_config.get('hsv_s', 0.7),
            hsv_v=train_config.get('hsv_v', 0.4),
            degrees=train_config.get('degrees', 0.0),
            translate=train_config.get('translate', 0.1),
            scale=train_config.get('scale', 0.5),
            fliplr=train_config.get('fliplr', 0.5),
            flipud=train_config.get('flipud', 0.0),
            mosaic=train_config.get('mosaic', 1.0),
            mixup=train_config.get('mixup', 0.0),
            copy_paste=train_config.get('copy_paste', 0.0),
            close_mosaic=train_config.get('close_mosaic', 10),
            amp=train_config.get('amp', True),
            cache=train_config.get('cache', False),
            seed=train_config.get('seed', 0),
        )
    
    # =========================================================================
    # TUNE 모드
    # =========================================================================
    elif mode == 'tune':
        tune_config = config.get('tune', {})
        data_yaml = tune_config.get('data', config.get('train', {}).get('data', 'dataset.yaml'))
        
        # 탐색 공간 파싱
        space_config = tune_config.get('space', None)
        space = parse_tune_space(space_config) if space_config else None
        
        print(f"\n하이퍼파라미터 튜닝 시작: {data_yaml}")
        
        trainer.tune(
            data_yaml=data_yaml,
            epochs=tune_config.get('epochs', 30),
            iterations=tune_config.get('iterations', 100),
            imgsz=tune_config.get('imgsz', 640),
            batch=tune_config.get('batch', 16),
            device=tune_config.get('device', None),
            workers=tune_config.get('workers', 8),
            optimizer=tune_config.get('optimizer', 'AdamW'),
            project=tune_config.get('project', 'runs/tune'),
            name=tune_config.get('name', None),
            exist_ok=tune_config.get('exist_ok', True),
            space=space,
            use_ray=tune_config.get('use_ray', False),
            gpu_per_trial=tune_config.get('gpu_per_trial', None),
            max_samples=tune_config.get('max_samples', 10),
            grace_period=tune_config.get('grace_period', 10),
            plots=tune_config.get('plots', True),
            save=tune_config.get('save', False),
            val=tune_config.get('val', False),
            seed=tune_config.get('seed', 0),
        )
    
    # =========================================================================
    # VALIDATE 모드
    # =========================================================================
    elif mode == 'validate':
        val_config = config.get('validate', {})
        data_yaml = val_config.get('data', 'data/merged_dataset/dataset.yaml')
        
        print(f"\n검증 시작: {data_yaml}")
        trainer.validate(
            data_yaml=data_yaml,
            split=val_config.get('split', 'val'),
            save_json=val_config.get('save_json', True),
            imgsz=val_config.get('imgsz', 640),
            batch=val_config.get('batch', 16),
            conf=val_config.get('conf', 0.001),
            iou=val_config.get('iou', 0.6),
            plots=val_config.get('plots', True),
        )
    
    # =========================================================================
    # PREDICT 모드
    # =========================================================================
    elif mode == 'predict':
        predict_config = config.get('predict', {})
        source = predict_config.get('source', 'data/merged_dataset/images/test')
        
        print(f"\n추론 시작: {source}")
        trainer.predict(
            source=source,
            conf=predict_config.get('conf', 0.25),
            iou=predict_config.get('iou', 0.45),
            imgsz=predict_config.get('imgsz', 640),
            save=predict_config.get('save', True),
            save_txt=predict_config.get('save_txt', False),
            save_conf=predict_config.get('save_conf', True),
            max_det=predict_config.get('max_det', 300),
        )
    
    # =========================================================================
    # EXPORT 모드
    # =========================================================================
    elif mode == 'export':
        export_config = config.get('export', {})
        format = export_config.get('format', 'onnx')
        
        print(f"\n모델 내보내기: {format}")
        trainer.export(
            format=format,
            imgsz=export_config.get('imgsz', 640),
            half=export_config.get('half', False),
            int8=export_config.get('int8', False),
            dynamic=export_config.get('dynamic', False),
            simplify=export_config.get('simplify', True),
        )
