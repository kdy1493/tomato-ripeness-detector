#!/usr/bin/env python3
"""
YOLO + DeepSORT 기본 트래킹

동영상 파일 · 웹캠 모두 지원

참고:
  - DeepSORT: https://github.com/levan92/deep_sort_realtime
  - Supervision: https://supervision.roboflow.com/latest/how_to/track_objects/
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_TRACKERS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(_TRACKERS_DIR))

from roi_utils import compute_roi, yolo_detections_with_roi

# ============================================================
# 설정  ← 이것만 수정하면 됨
# ============================================================
CONFIG = {
    # 입출력
    "source":      "notebook/rgb.mp4",
    "model_path":  "runs/yolo26_custom_tomato/trained_yolo26_custom.pt",
    "output_path": "tracking_result/basic_deepsort.mp4",
    "show_window": True,

    # Detection
    "conf": 0.5,
    "iou":  0.3,

    # ROI (src/tracker.py 와 동일). None 또는 0 이면 전체 프레임
    "roi_half_width": 320,

    # DeepSORT
    "max_age":         30,
    "n_init":           3,
    "max_cosine_dist":  0.3,
    "nn_budget":       100,

    # 시각화
    "show_trace":   True,
    "trace_length": 30,
}
# ============================================================

CLASS_NAMES = {0: "ripe", 1: "unripe"}
COLORS      = {0: (0, 80, 255), 1: (0, 200, 80)}


def get_model(model_path: str) -> YOLO:
    p = REPO_ROOT / model_path
    if p.exists():
        return YOLO(str(p))
    if Path(model_path).exists():
        return YOLO(model_path)
    candidates = sorted((REPO_ROOT / "models").glob("*.pt")) if (REPO_ROOT / "models").exists() else []
    if candidates:
        return YOLO(str(candidates[-1]))
    return YOLO("yolov8n.pt")


def open_source(source: str) -> cv2.VideoCapture:
    if str(source).isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(str(REPO_ROOT / source))
    if not cap.isOpened():
        raise RuntimeError(f"영상 소스를 열 수 없습니다: {source}")
    return cap


def make_writer(output_path: Optional[str], width: int, height: int, fps: float):
    if not output_path:
        return None
    out = REPO_ROOT / output_path
    out.parent.mkdir(parents=True, exist_ok=True)
    return cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))


def make_deepsort(config: dict) -> DeepSort:
    return DeepSort(
        max_age=config["max_age"],
        n_init=config["n_init"],
        max_cosine_distance=config["max_cosine_dist"],
        nn_budget=config["nn_budget"],
        embedder="mobilenet",
        half=True,
        bgr=True,
    )


def draw_track(frame, x1, y1, x2, y2, track_id, class_id, conf):
    color = COLORS.get(class_id, (255, 200, 0))
    label = f"{CLASS_NAMES.get(class_id, class_id)} #{track_id}  {conf:.2f}"
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 2, y1), color, -1)
    cv2.putText(frame, label, (x1 + 1, y1 - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)


def run(config: dict) -> dict:
    """DeepSORT 트래킹 실행.

    Returns:
        dict:
            mot_rows    (List[tuple]): (frame_id, track_id, x, y, w, h, conf, class_id)
            fps_avg     (float)
            total_frames (int)
            unique_ids  (Dict[int, set]): {class_id: set of track_ids}
    """
    model = get_model(config["model_path"])
    cap   = open_source(config["source"])
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    roi   = compute_roi(w, h, config.get("roi_half_width"))
    if roi is not None:
        print(f"[DeepSORT] ROI x=[{roi[0]}, {roi[2]}] (roi_half_width={config.get('roi_half_width')})")

    trackers:      Dict[int, DeepSort]        = {cid: make_deepsort(config) for cid in CLASS_NAMES}
    # raw tracker ID → stable ID (클래스별, cy-cx 오름차순으로 발급)
    raw_to_stable: Dict[int, Dict[int, int]] = {cid: {} for cid in CLASS_NAMES}
    next_sid:      Dict[int, int]            = {cid: 1 for cid in CLASS_NAMES}
    seen_ids:      Dict[int, set]            = {cid: set() for cid in CLASS_NAMES}
    mot_rows:      List[Tuple]               = []
    traces:        dict                      = {}

    writer      = make_writer(config.get("output_path"), w, h, fps)
    show_window = config.get("show_window", False)
    show_trace  = config.get("show_trace", False)
    trace_len   = config.get("trace_length", 30)

    frame_idx  = 0
    fps_acc    = 0.0
    active_ids = set()

    print(f"[DeepSORT] 시작...")

    while True:
        t0 = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        active_ids.clear()

        all_dets = yolo_detections_with_roi(
            model, frame, config["conf"], config["iou"], roi,
        )

        cls_dets: Dict[int, list] = {cid: [] for cid in CLASS_NAMES}
        if len(all_dets) > 0 and all_dets.class_id is not None:
            for i in range(len(all_dets)):
                cid = int(all_dets.class_id[i])
                if cid not in CLASS_NAMES:
                    continue
                x1, y1, x2, y2 = all_dets.xyxy[i].tolist()
                conf_v = float(all_dets.confidence[i])
                cls_dets[cid].append(([x1, y1, x2 - x1, y2 - y1], conf_v, cid))

        for cid, tracker in trackers.items():
            tracks = tracker.update_tracks(cls_dets[cid], frame=frame)
            confirmed = [t for t in tracks if t.is_confirmed()]
            if not confirmed:
                continue

            # tracker.py 방식: 신규 raw ID를 cy-cx 오름차순(우상단→좌하단)으로 정렬 후 순번 발급
            raw_ids = np.array([t.track_id for t in confirmed])
            ltrbs   = np.array([t.to_ltrb() for t in confirmed])
            new_idx = [i for i, r in enumerate(raw_ids) if int(r) not in raw_to_stable[cid]]
            new_idx.sort(key=lambda i: (
                0.5 * (ltrbs[i][1] + ltrbs[i][3])   # cy
                - 0.5 * (ltrbs[i][0] + ltrbs[i][2])  # cx
            ))
            for i in new_idx:
                raw_to_stable[cid][int(raw_ids[i])] = next_sid[cid]
                next_sid[cid] += 1
            stable_ids = np.array([raw_to_stable[cid][int(r)] for r in raw_ids], dtype=np.int64)

            for i, t in enumerate(confirmed):
                tid    = int(stable_ids[i])
                x1, y1, x2, y2 = map(int, ltrbs[i])
                cx, cy  = (x1 + x2) // 2, (y1 + y2) // 2
                conf_v  = t.det_conf if t.det_conf is not None else 0.0

                seen_ids[cid].add(tid)
                active_ids.add(tid)
                mot_rows.append((
                    frame_idx, tid,
                    x1, y1, x2 - x1, y2 - y1,
                    float(conf_v), cid,
                ))

                if show_trace:
                    traces.setdefault(tid, []).append((cx, cy))
                    traces[tid] = traces[tid][-trace_len:]

                if writer or show_window:
                    draw_track(frame, x1, y1, x2, y2, tid, cid, conf_v)

        # 비활성 궤적 정리
        for tid in list(traces.keys()):
            if tid not in active_ids:
                traces.pop(tid, None)

        elapsed = time.perf_counter() - t0
        fps_acc = 0.1 * (1 / elapsed if elapsed > 0 else 0) + 0.9 * fps_acc if fps_acc else (1 / elapsed if elapsed > 0 else 0)

        if writer or show_window:
            if show_trace:
                for pts in traces.values():
                    for i in range(1, len(pts)):
                        cv2.line(frame, pts[i - 1], pts[i], (180, 180, 180), 1)

            if roi is not None:
                cv2.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]), (255, 200, 0), 2)

            for i, text in enumerate([
                f"Frame {frame_idx}",
                f"ripe   total: {len(seen_ids[0])}",
                f"unripe total: {len(seen_ids[1])}",
            ]):
                y = 30 + i * 28
                cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 3)
                cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1)

            if writer:
                writer.write(frame)
            if show_window:
                cv2.imshow("YOLO + DeepSORT", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

    cap.release()
    if writer:
        writer.release()
    if show_window:
        cv2.destroyAllWindows()

    print(f"[DeepSORT] 완료 | {frame_idx}프레임 | FPS={fps_acc:.1f} | "
          f"ripe={len(seen_ids[0])} unripe={len(seen_ids[1])}")

    return {
        "mot_rows":     mot_rows,
        "fps_avg":      fps_acc,
        "total_frames": frame_idx,
        "unique_ids":   seen_ids,
    }


def main() -> None:
    result = run(CONFIG)
    print(f"[결과] ripe {len(result['unique_ids'][0])}개 / unripe {len(result['unique_ids'][1])}개 (누적 고유 ID 기준)")


if __name__ == "__main__":
    main()
