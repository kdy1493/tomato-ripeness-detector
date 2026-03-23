"""YOLO26 model trainer and utilities."""

import yaml
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple

from ultralytics import YOLO


AVAILABLE_MODELS = {
    "nano": "yolo26n.pt",
    "small": "yolo26s.pt",
    "medium": "yolo26m.pt",
    "large": "yolo26l.pt",
    "xlarge": "yolo26x.pt",
}

TASK_SUFFIXES = {
    "segment": "-seg",
    "pose": "-pose",
    "obb": "-obb",
    "classify": "-cls",
}


class YOLO26Trainer:
    """YOLO26 model training, tuning, validation, prediction, and export."""

    def __init__(self, model_size: str = "nano", task: str = "detect",
                 pretrained: bool = True):
        self.model_size = model_size
        self.task = task
        self.pretrained = pretrained
        self.model = None
        self.results = None

        base = AVAILABLE_MODELS.get(model_size, "yolo26n.pt")
        if task != "detect":
            suffix = TASK_SUFFIXES.get(task, "")
            base = base.replace(".pt", f"{suffix}.pt")
        self.model_name = base

    def load_model(self):
        if self.pretrained:
            self.model = YOLO(self.model_name)
        else:
            self.model = YOLO(self.model_name.replace(".pt", ".yaml"))
        return self.model

    def _auto_device(self, device):
        if device is None:
            return "0" if torch.cuda.is_available() else "cpu"
        return device

    def _auto_name(self, prefix: str, name=None):
        if name is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            return f"{prefix}_{self.model_size}_{ts}"
        return name

    def train(self, data_yaml: str, **kwargs):
        if self.model is None:
            self.load_model()
        kwargs.setdefault("device", self._auto_device(kwargs.get("device")))
        kwargs.setdefault("name", self._auto_name("yolo26", kwargs.get("name")))
        self.results = self.model.train(data=data_yaml, **kwargs)
        return self.results

    def tune(self, data_yaml: str, space: Dict = None, **kwargs):
        if self.model is None:
            self.load_model()
        kwargs.setdefault("device", self._auto_device(kwargs.get("device")))
        kwargs.setdefault("name", self._auto_name("tune", kwargs.get("name")))

        if space is None:
            space = {
                "lr0": (1e-5, 1e-1),
                "lrf": (0.01, 1.0),
                "momentum": (0.6, 0.98),
                "weight_decay": (0.0, 0.001),
                "box": (0.02, 0.2),
                "cls": (0.2, 4.0),
            }

        use_ray = kwargs.pop("use_ray", False)
        if use_ray:
            try:
                from ray import tune as ray_tune
                space = {k: ray_tune.uniform(*v) for k, v in space.items()}
                kwargs["use_ray"] = True
            except ImportError:
                print("Ray Tune not installed, falling back to genetic algorithm.")
                use_ray = False

        return self.model.tune(data=data_yaml, space=space, **kwargs)

    def validate(self, data_yaml: str = None, **kwargs):
        if self.model is None:
            self.load_model()
        return self.model.val(data=data_yaml, **kwargs)

    def predict(self, source, **kwargs):
        if self.model is None:
            self.load_model()
        return self.model.predict(source=source, **kwargs)

    def export(self, format: str = "onnx", **kwargs):
        if self.model is None:
            self.load_model()
        return self.model.export(format=format, **kwargs)


def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_tune_space(space_config: Dict) -> Dict[str, Tuple[float, float]]:
    parsed = {}
    for key, value in space_config.items():
        if isinstance(value, (list, tuple)) and len(value) == 2:
            parsed[key] = tuple(value)
    return parsed
