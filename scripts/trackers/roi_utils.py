"""ROI + YOLO 검출 (src/tracker.py 의 run / run_benchmark 와 동일한 방식)."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import supervision as sv
from ultralytics import YOLO


def compute_roi(
    w: int, h: int, roi_half_width: Optional[int],
) -> Optional[Tuple[int, int, int, int]]:
    """프레임 중앙 기준 가로 스트립. roi_half_width 가 None 또는 0 이면 전체 프레임."""
    if not roi_half_width:
        return None
    cx = w // 2
    return (max(0, cx - roi_half_width), 0, min(w - 1, cx + roi_half_width), h - 1)


def yolo_detections_with_roi(
    model: YOLO,
    frame: np.ndarray,
    conf: float,
    nms: float,
    roi: Optional[Tuple[int, int, int, int]],
) -> sv.Detections:
    """ROI 크롭 검출 후 xyxy 를 전역 좌표로 복원. roi 가 없으면 전체 프레임."""
    if roi is not None:
        x0, y0, x1, y1 = roi
        crop = frame[y0 : y1 + 1, x0 : x1 + 1]
        if crop.size == 0:
            return sv.Detections.empty()
        res = model(crop, conf=conf, verbose=False)[0]
        dets = sv.Detections.from_ultralytics(res).with_nms(threshold=nms)
        if len(dets) > 0:
            off = np.array([x0, y0, x0, y0], dtype=np.float32)
            dets.xyxy = dets.xyxy + off
        return dets
    res = model(frame, conf=conf, verbose=False)[0]
    return sv.Detections.from_ultralytics(res).with_nms(threshold=nms)
