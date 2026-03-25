"""Real-time object tracking with YOLO26 + ByteTrack."""

import time
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from trackers import ByteTrackTracker

CLASS_NAMES = {0: "ripe", 1: "unripe"}
CLASS_COLORS = sv.ColorPalette.from_hex([
    "#FF0000",  # class 0: ripe → red
    "#00CC00",  # class 1: unripe → green
])

DEFAULT_MODEL = r"runs\yolo26_merged_tomato\weights\epoch80.pt"


@dataclass
class _TrackState:
    stable_id: int
    xyxy: np.ndarray
    class_id: int
    hist: Optional[np.ndarray]


class StableIdAssigner:
    """ByteTrack ID와 무관하게 화면 이탈 후에도 동일한 표시 ID를 유지.

    - 연속 프레임: 같은 클래스 + IoU
    - 재등장: HSV 히스토그램 상관 (옵션: 화면상 bbox 중심 거리 가중, 고정 카메라에 유리)
    """

    def __init__(
        self,
        lost_ttl_frames: int = 900,
        iou_match_thresh: float = 0.2,
        hist_match_thresh: float = 0.62,
        hist_margin: float = 0.06,
        hist_ema: float = 0.35,
        use_position_score: bool = True,
        position_sigma_px: float = 200.0,
        combined_hist_weight: float = 0.65,
        combined_match_thresh: float = 0.55,
    ) -> None:
        self.lost_ttl_frames = lost_ttl_frames
        self.iou_match_thresh = iou_match_thresh
        self.hist_match_thresh = hist_match_thresh
        self.hist_margin = hist_margin
        self.hist_ema = hist_ema
        self.use_position_score = use_position_score
        self.position_sigma_px = max(1e-3, position_sigma_px)
        wh = float(combined_hist_weight)
        self._hist_w = wh
        self._pos_w = max(0.0, min(1.0, 1.0 - wh))
        s = self._hist_w + self._pos_w
        if s > 1e-6:
            self._hist_w /= s
            self._pos_w /= s
        self.combined_match_thresh = combined_match_thresh
        self._next_stable_id = 1
        self._prev: List[_TrackState] = []
        self._lost: List[dict] = []

    def reset(self) -> None:
        self._next_stable_id = 1
        self._prev = []
        self._lost = []

    def _new_id(self) -> int:
        sid = self._next_stable_id
        self._next_stable_id += 1
        return sid

    @staticmethod
    def _bbox_hist(frame: np.ndarray, xyxy: np.ndarray) -> Optional[np.ndarray]:
        h, w = frame.shape[:2]
        x1 = int(max(0, xyxy[0]))
        y1 = int(max(0, xyxy[1]))
        x2 = int(min(w, xyxy[2]))
        y2 = int(min(h, xyxy[3]))
        if x2 - x1 < 4 or y2 - y1 < 4:
            return None
        crop = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [18, 12], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist

    @staticmethod
    def _iou(a: np.ndarray, b: np.ndarray) -> float:
        return float(sv.box_iou_batch(a.reshape(1, 4), b.reshape(1, 4))[0, 0])

    @staticmethod
    def _position_score(
        lost_xyxy: np.ndarray, cur_xyxy: np.ndarray, sigma_px: float,
    ) -> float:
        """중심점 거리 기반 0~1. sigma_px에서 0에 가깝게."""
        lax = 0.5 * (float(lost_xyxy[0]) + float(lost_xyxy[2]))
        lay = 0.5 * (float(lost_xyxy[1]) + float(lost_xyxy[3]))
        cax = 0.5 * (float(cur_xyxy[0]) + float(cur_xyxy[2]))
        cay = 0.5 * (float(cur_xyxy[1]) + float(cur_xyxy[3]))
        dist = float(np.hypot(lax - cax, lay - cay))
        return max(0.0, 1.0 - dist / sigma_px)

    def assign(self, frame_idx: int, frame: np.ndarray, d: sv.Detections) -> np.ndarray:
        n = len(d)
        if n == 0:
            for p in self._prev:
                self._lost.append({
                    "stable_id": p.stable_id,
                    "class_id": p.class_id,
                    "hist": p.hist,
                    "last_xyxy": p.xyxy.copy(),
                    "deadline": frame_idx + self.lost_ttl_frames,
                })
            self._prev = []
            self._lost = [x for x in self._lost if x["deadline"] > frame_idx]
            return np.array([], dtype=np.int32)

        xyxy = d.xyxy
        classes = d.class_id.astype(np.int32)
        hists = [self._bbox_hist(frame, xyxy[i]) for i in range(n)]

        stable = np.full(n, -1, dtype=np.int32)
        matched_prev_idx: set[int] = set()

        for pi, p in enumerate(self._prev):
            best_j = -1
            best_iou = 0.0
            for j in range(n):
                if stable[j] != -1:
                    continue
                if int(classes[j]) != p.class_id:
                    continue
                iou = self._iou(p.xyxy, xyxy[j])
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_j >= 0 and best_iou >= self.iou_match_thresh:
                stable[best_j] = p.stable_id
                matched_prev_idx.add(pi)

        for pi, p in enumerate(self._prev):
            if pi not in matched_prev_idx:
                self._lost.append({
                    "stable_id": p.stable_id,
                    "class_id": p.class_id,
                    "hist": p.hist,
                    "last_xyxy": p.xyxy.copy(),
                    "deadline": frame_idx + self.lost_ttl_frames,
                })

        self._lost = [x for x in self._lost if x["deadline"] > frame_idx]

        match_thresh = (
            self.combined_match_thresh
            if self.use_position_score
            else self.hist_match_thresh
        )

        for j in range(n):
            if stable[j] != -1:
                continue
            hj = hists[j]
            if hj is None:
                stable[j] = self._new_id()
                continue
            best_k = -1
            best_s = -1.0
            second_s = -1.0
            for k, L in enumerate(self._lost):
                if L["class_id"] != int(classes[j]):
                    continue
                lh = L["hist"]
                if lh is None:
                    continue
                hist_s = max(0.0, float(cv2.compareHist(hj, lh, cv2.HISTCMP_CORREL)))
                if self.use_position_score:
                    pos_s = self._position_score(
                        L["last_xyxy"], xyxy[j], self.position_sigma_px,
                    )
                    s = self._hist_w * hist_s + self._pos_w * pos_s
                else:
                    s = hist_s
                if s > best_s:
                    second_s = best_s
                    best_s = s
                    best_k = k
                elif s > second_s:
                    second_s = s
            if (
                best_k >= 0
                and best_s >= match_thresh
                and (best_s - second_s) >= self.hist_margin
            ):
                L = self._lost.pop(best_k)
                stable[j] = int(L["stable_id"])
            else:
                stable[j] = self._new_id()

        new_prev: List[_TrackState] = []
        for j in range(n):
            sid = int(stable[j])
            old_hist = None
            for p in self._prev:
                if p.stable_id == sid:
                    old_hist = p.hist
                    break
            hj = hists[j]
            if hj is not None and old_hist is not None:
                nh = (1.0 - self.hist_ema) * old_hist + self.hist_ema * hj
            elif hj is not None:
                nh = hj
            else:
                nh = old_hist
            new_prev.append(
                _TrackState(
                    stable_id=sid,
                    xyxy=xyxy[j].copy(),
                    class_id=int(classes[j]),
                    hist=nh,
                )
            )
        self._prev = new_prev
        return stable


def build_annotators():
    box = sv.BoxAnnotator(
        color=CLASS_COLORS,
        color_lookup=sv.ColorLookup.CLASS,
    )
    label = sv.LabelAnnotator(
        color=CLASS_COLORS,
        color_lookup=sv.ColorLookup.CLASS,
        text_color=sv.Color.WHITE,
        text_scale=0.6,
    )
    trace = sv.TraceAnnotator(
        color=CLASS_COLORS,
        color_lookup=sv.ColorLookup.CLASS,
        thickness=2,
        trace_length=100,
    )
    return box, label, trace


def build_motion_compensator():
    from trackers import MotionEstimator, MotionAwareTraceAnnotator

    estimator = MotionEstimator(
        max_points=500,
        min_distance=10,
        quality_level=0.001,
        ransac_reproj_threshold=1.0,
    )
    motion_trace = MotionAwareTraceAnnotator(
        color=CLASS_COLORS,
        color_lookup=sv.ColorLookup.CLASS,
        thickness=2,
        trace_length=100,
    )
    return estimator, motion_trace


def run(*, source="0", model_path=DEFAULT_MODEL, conf=0.2, nms=0.3,
        compensation=False, min_frames=1, stable_ids=True,
        lost_ttl_frames=900, byte_lost_buffer=120, hist_match_thresh=0.62,
        use_position_reid: bool = True, position_sigma_px: float = 200.0,
        combined_hist_weight: float = 0.65, combined_match_thresh: float = 0.55):
    """Run real-time tracking loop.

    Args:
        source: Webcam index (str/int) or RTSP/file path.
        model_path: Path to YOLO weights.
        conf: Detection confidence threshold.
        nms: NMS IoU threshold.
        compensation: Enable Camera Motion Compensation.
        min_frames: Minimum consecutive frames before confirming a track.
        stable_ids: If True, keep display IDs after leave/re-enter (IoU + HSV hist).
        lost_ttl_frames: How long (frames) to remember IDs for re-match.
        byte_lost_buffer: ByteTrack internal buffer (longer = better short occlusion).
        hist_match_thresh: HSV-only re-match threshold (when use_position_reid False).
        use_position_reid: Blend bbox center distance with hist (good for fixed camera).
        position_sigma_px: Distance at which position score nears 0.
        combined_hist_weight: Weight of hist in combined score (rest is position).
        combined_match_thresh: Min combined score when use_position_reid True.
    """
    model = YOLO(model_path)
    vid_source = int(source) if isinstance(source, str) and source.isdigit() else source
    cap = cv2.VideoCapture(vid_source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")
    fps_cap = cap.get(cv2.CAP_PROP_FPS)
    fps_hint = float(fps_cap) if fps_cap and fps_cap > 1.0 else 30.0

    tracker = ByteTrackTracker(
        minimum_consecutive_frames=min_frames,
        lost_track_buffer=byte_lost_buffer,
        frame_rate=fps_hint,
    )
    id_assigner = StableIdAssigner(
        lost_ttl_frames=lost_ttl_frames,
        hist_match_thresh=hist_match_thresh,
        use_position_score=use_position_reid,
        position_sigma_px=position_sigma_px,
        combined_hist_weight=combined_hist_weight,
        combined_match_thresh=combined_match_thresh,
    ) if stable_ids else None
    box_ann, label_ann, trace_ann = build_annotators()

    motion_estimator = None
    motion_trace_ann = None
    if compensation:
        motion_estimator, motion_trace_ann = build_motion_compensator()
        print("[INFO] Camera Motion Compensation ENABLED")
    else:
        print("[INFO] Camera Motion Compensation DISABLED (fixed camera mode)")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Source: {source} ({w}x{h})")
    print(f"[INFO] Model: {model_path}")
    print(f"[INFO] Confidence: {conf} / NMS: {nms}")
    if id_assigner is not None:
        if use_position_reid:
            print(
                f"[INFO] Stable IDs ON (HSV+pos), ttl~{lost_ttl_frames}f, "
                f"combined>={combined_match_thresh}, "
                f"hist_w={combined_hist_weight:.2f}, sigma={position_sigma_px}px",
            )
        else:
            print(
                f"[INFO] Stable IDs ON (HSV only), ttl~{lost_ttl_frames}f, "
                f"corr>={hist_match_thresh}",
            )
    else:
        print("[INFO] Stable IDs OFF (ByteTrack IDs only)")
    print("[INFO] Press 'q' to quit, 'r' to reset tracker, 'c' to toggle compensation")

    fps_avg = 0.0
    alpha = 0.1
    frame_idx = 0

    while True:
        t0 = time.perf_counter()

        ret, frame = cap.read()
        if not ret:
            break

        coord_transform = None
        if motion_estimator is not None:
            try:
                coord_transform = motion_estimator.update(frame)
            except Exception:
                coord_transform = None

        results = model(frame, conf=conf, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results).with_nms(
            threshold=nms,
        )
        detections = tracker.update(detections)

        # ripe(0), unripe(1)만 남기기
        valid = [i for i, c in enumerate(detections.class_id)
                 if c in CLASS_NAMES] if detections.class_id is not None else []
        detections = detections[valid]

        if id_assigner is not None:
            stable = id_assigner.assign(frame_idx, frame, detections)
            detections.tracker_id = stable

        n_ripe = int((detections.class_id == 0).sum()) if detections.class_id is not None else 0
        n_unripe = int((detections.class_id == 1).sum()) if detections.class_id is not None else 0

        annotated = frame.copy()
        annotated = box_ann.annotate(annotated, detections)

        if motion_trace_ann is not None and coord_transform is not None:
            annotated = motion_trace_ann.annotate(
                annotated, detections, coord_transform=coord_transform,
            )
        else:
            annotated = trace_ann.annotate(annotated, detections)

        labels = []
        for i in range(len(detections)):
            cls = detections.class_id[i] if detections.class_id is not None else -1
            tid = detections.tracker_id[i] if detections.tracker_id is not None else -1
            name = CLASS_NAMES.get(int(cls), "?")
            labels.append(f"{name} #{int(tid)}")
        annotated = label_ann.annotate(annotated, detections, labels)

        dt = time.perf_counter() - t0
        fps = 1.0 / dt if dt > 0 else 0
        fps_avg = alpha * fps + (1 - alpha) * fps_avg if fps_avg > 0 else fps

        cv2.putText(
            annotated,
            f"Ripe: {n_ripe} | Unripe: {n_unripe} | FPS: {fps_avg:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        cv2.imshow("YOLO26 Real-time Tracking", annotated)

        frame_idx += 1

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            tracker.reset()
            if id_assigner is not None:
                id_assigner.reset()
            if motion_estimator is not None:
                motion_estimator.reset()
            frame_idx = 0
            print("[INFO] Tracker reset")
        elif key == ord("c"):
            if motion_estimator is None:
                motion_estimator, motion_trace_ann = build_motion_compensator()
                print("[INFO] Compensation ON")
            else:
                motion_estimator = None
                motion_trace_ann = None
                print("[INFO] Compensation OFF")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Done.")


if __name__ == "__main__":
    run()
