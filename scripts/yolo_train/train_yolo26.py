"""
YOLO26 학습 실행 스크립트.

사용법:
    python scripts/yolo_train/train_yolo26.py -c config/train_yolo26.yaml
    python scripts/yolo_train/train_yolo26.py -c config/train_yolo26.yaml --mode tune
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.trainer import YOLO26Trainer, load_config, parse_tune_space


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO26 학습")
    parser.add_argument(
        "-c",
        "--config",
        default=str(REPO_ROOT / "config" / "train_yolo26.yaml"),
        help="학습 설정 YAML 파일 경로",
    )
    parser.add_argument("--mode", default=None,
                        choices=["train", "tune", "validate", "predict", "export"],
                        help="실행 모드 (config보다 우선)")
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"Config loaded: {args.config}")

    model_cfg = config.get("model", {})
    trainer = YOLO26Trainer(
        model_size=model_cfg.get("size", "nano"),
        task=model_cfg.get("task", "detect"),
        pretrained=model_cfg.get("pretrained", True),
    )
    trainer.load_model()

    mode = args.mode or config.get("mode", "train")

    if mode == "train":
        cfg = config.get("train", {})
        data = cfg.pop("data", "data/merged_dataset/dataset.yaml")
        trainer.train(data_yaml=data, **cfg)

    elif mode == "tune":
        cfg = config.get("tune", {})
        data = cfg.pop("data", config.get("train", {}).get("data", "dataset.yaml"))
        space_raw = cfg.pop("space", None)
        space = parse_tune_space(space_raw) if space_raw else None
        trainer.tune(data_yaml=data, space=space, **cfg)

    elif mode == "validate":
        cfg = config.get("validate", {})
        data = cfg.pop("data", "data/merged_dataset/dataset.yaml")
        trainer.validate(data_yaml=data, **cfg)

    elif mode == "predict":
        cfg = config.get("predict", {})
        source = cfg.pop("source", "data/merged_dataset/images/test")
        trainer.predict(source=source, **cfg)

    elif mode == "export":
        cfg = config.get("export", {})
        fmt = cfg.pop("format", "onnx")
        trainer.export(format=fmt, **cfg)


if __name__ == "__main__":
    main()
