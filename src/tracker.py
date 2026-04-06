"""Real-time object tracking with YOLO26 + ByteTrack."""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import supervision as sv
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO
from trackers import ByteTrackTracker

CLASS_NAMES = {0: "ripe", 1: "unripe"}
CLASS_COLORS = sv.ColorPalette.from_hex([
    "#FF0000",  # class 0: ripe → red
    "#00CC00",  # class 1: unripe → green
])

DEFAULT_MODEL = r"runs\yolo26_custom_tomato\trained_yolo26_custom.pt"

# 터미널에서 새 로직 적용 여부 확인용 (스크립트 시작 시 출력)
STABLE_ID_LOGIC_TAG = "hungarian+global-gap+reid-relax"


@dataclass
class _TrackState:
    stable_id: int
    xyxy: np.ndarray
    class_id: int
    hist: Optional[np.ndarray]


class StableIdAssigner:
    """ByteTrack ID와 무관하게 화면 이탈 후에도 동일한 표시 ID를 유지.

    - 연속 프레임: 클래스별 IoU 1:1 최적 매칭(헝가리안) + IoU 부족 시 가까운 중심거리 보조
    - 재등장: HSV 히스토그램 상관 (옵션: 화면상 bbox 중심 거리 가중, 고정 카메라에 유리)
    - 표시 ID는 클래스마다 1부터 따로 증가 (ripe #1 과 unripe #1 동시 존재 가능).
    - 같은 프레임에서 새로 발급하는 번호만, 클래스별 우상단→좌하단(위→아래, 같은 줄은 오→왼) 순으로 부여.
    """

    def __init__(
        self,
        lost_ttl_frames: int = 900,
        iou_match_thresh: float = 0.12,
        hist_match_thresh: float = 0.62,
        hist_margin: float = 0.06,
        hist_ema: float = 0.35,
        use_position_score: bool = True,
        position_sigma_px: float = 200.0,
        combined_hist_weight: float = 0.65,
        combined_match_thresh: float = 0.55,
        center_match_max_px: float = 130.0,
        recent_reid_max_frames: int = 40,
        recent_reid_max_center_dist_px: float = 180.0,
        recent_reid_min_hist: float = 0.42,
        gap_position_only_frames: int = 55,
        gap_position_only_max_px: float = 95.0,
        gap_position_extra_per_frame: float = 11.0,
        gap_position_only_cap_px: float = 320.0,
    ) -> None:
        self.lost_ttl_frames = lost_ttl_frames
        self.iou_match_thresh = iou_match_thresh
        self.hist_match_thresh = hist_match_thresh
        self.hist_margin = hist_margin
        self.hist_ema = hist_ema
        self.use_position_score = use_position_score
        self.position_sigma_px = max(1e-3, position_sigma_px)
        self.center_match_max_px = max(1e-3, center_match_max_px)
        self.recent_reid_max_frames = max(1, int(recent_reid_max_frames))
        self.recent_reid_max_center_dist_px = max(1e-3, recent_reid_max_center_dist_px)
        self.recent_reid_min_hist = recent_reid_min_hist
        self.gap_position_only_frames = max(1, int(gap_position_only_frames))
        self.gap_position_only_max_px = max(1e-3, gap_position_only_max_px)
        self.gap_position_extra_per_frame = max(0.0, gap_position_extra_per_frame)
        self.gap_position_only_cap_px = max(
            self.gap_position_only_max_px, gap_position_only_cap_px,
        )
        wh = float(combined_hist_weight)
        self._hist_w = wh
        self._pos_w = max(0.0, min(1.0, 1.0 - wh))
        s = self._hist_w + self._pos_w
        if s > 1e-6:
            self._hist_w /= s
            self._pos_w /= s
        self.combined_match_thresh = combined_match_thresh
        self._next_id_by_class: Dict[int, int] = {}
        self._prev: List[_TrackState] = []
        self._lost: List[dict] = []

    def config_log_line(self) -> str:
        return (
            f"[INFO] {STABLE_ID_LOGIC_TAG}: IoU≥{self.iou_match_thresh}, "
            f"prev-center≤{self.center_match_max_px}px, "
            f"gap≤{self.gap_position_only_frames}f base≤{self.gap_position_only_max_px}px "
            f"+{self.gap_position_extra_per_frame}/f cap{self.gap_position_only_cap_px}px"
        )

    def reset(self) -> None:
        self._next_id_by_class.clear()
        self._prev = []
        self._lost = []

    def _new_id(self, class_id: int) -> int:
        n = self._next_id_by_class.get(class_id, 1)
        self._next_id_by_class[class_id] = n + 1
        return n

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
    def _center_dist(a_xyxy: np.ndarray, b_xyxy: np.ndarray) -> float:
        ax = 0.5 * (float(a_xyxy[0]) + float(a_xyxy[2]))
        ay = 0.5 * (float(a_xyxy[1]) + float(a_xyxy[3]))
        bx = 0.5 * (float(b_xyxy[0]) + float(b_xyxy[2]))
        by = 0.5 * (float(b_xyxy[1]) + float(b_xyxy[3]))
        return float(np.hypot(ax - bx, ay - by))

    def _match_prev_iou_hungarian(
        self,
        n: int,
        xyxy: np.ndarray,
        classes: np.ndarray,
        stable: np.ndarray,
        matched_prev_idx: set[int],
    ) -> None:
        """클래스별로 IoU 합이 최대가 되도록 1:1 매칭 (가까이 붙은 토마토 ID 뒤바뀜 완화)."""
        class_ids = {int(c) for c in classes}
        class_ids |= {p.class_id for p in self._prev}
        for c in sorted(class_ids):
            p_idxs = [pi for pi, p in enumerate(self._prev) if p.class_id == c]
            j_idxs = [j for j in range(n) if int(classes[j]) == c]
            if not p_idxs or not j_idxs:
                continue
            n_p, n_j = len(p_idxs), len(j_idxs)
            cost = np.ones((n_p, n_j), dtype=np.float64)
            iou_mat = np.zeros((n_p, n_j), dtype=np.float64)
            for ii, pi in enumerate(p_idxs):
                for jj, j in enumerate(j_idxs):
                    iou_v = self._iou(self._prev[pi].xyxy, xyxy[j])
                    iou_mat[ii, jj] = iou_v
                    cost[ii, jj] = 1.0 - iou_v
            ri, ci = linear_sum_assignment(cost)
            for ii, jj in zip(ri, ci):
                if iou_mat[ii, jj] < self.iou_match_thresh:
                    continue
                j, pi = j_idxs[jj], p_idxs[ii]
                if stable[j] != -1:
                    continue
                stable[j] = self._prev[pi].stable_id
                matched_prev_idx.add(pi)

    def _match_prev_center_fallback(
        self,
        n: int,
        xyxy: np.ndarray,
        classes: np.ndarray,
        stable: np.ndarray,
        matched_prev_idx: set[int],
    ) -> None:
        """짧은 프레임에서 박스가 튀어 IoU만으로는 실패할 때, 같은 클래스·가까운 중심으로 1:1 보강."""
        pairs: List[Tuple[float, int, int]] = []
        for pi, p in enumerate(self._prev):
            if pi in matched_prev_idx:
                continue
            for j in range(n):
                if stable[j] != -1:
                    continue
                if int(classes[j]) != p.class_id:
                    continue
                d = self._center_dist(p.xyxy, xyxy[j])
                if d <= self.center_match_max_px:
                    pairs.append((d, pi, j))
        pairs.sort(key=lambda t: t[0])
        used_p: set[int] = set()
        used_j: set[int] = set()
        for d, pi, j in pairs:
            if pi in used_p or j in used_j:
                continue
            used_p.add(pi)
            used_j.add(j)
            stable[j] = self._prev[pi].stable_id
            matched_prev_idx.add(pi)

    def _fill_gap_position_matches(
        self,
        xyxy: np.ndarray,
        classes: np.ndarray,
        stable: np.ndarray,
        frame_idx: int,
        gap_candidates: List[int],
    ) -> None:
        """lost(최근 것 우선) ↔ 미매칭 검출을 거리 기준으로 전역 1:1 (검출 루프 순서로 #5를 #9에 뺏기지 않게)."""
        if not self._lost or not gap_candidates:
            return
        lost_order = sorted(
            range(len(self._lost)),
            key=lambda kk: -int(self._lost[kk].get("lost_at", -10**9)),
        )
        used_j: set[int] = set()
        remove_k: List[int] = []
        for k in lost_order:
            L = self._lost[k]
            lost_at = int(L.get("lost_at", -10**9))
            age = frame_idx - lost_at
            if age > self.gap_position_only_frames:
                continue
            max_d = min(
                self.gap_position_only_cap_px,
                self.gap_position_only_max_px
                + self.gap_position_extra_per_frame * max(0, age),
            )
            cid_l = int(L["class_id"])
            best_j = -1
            best_d = float("inf")
            for j in gap_candidates:
                if j in used_j or int(stable[j]) != -1:
                    continue
                if int(classes[j]) != cid_l:
                    continue
                dist = self._center_dist(L["last_xyxy"], xyxy[j])
                if dist <= max_d and dist < best_d:
                    best_d = dist
                    best_j = j
            if best_j >= 0:
                stable[best_j] = int(L["stable_id"])
                used_j.add(best_j)
                remove_k.append(k)
        for k in sorted(remove_k, reverse=True):
            self._lost.pop(k)

    @staticmethod
    def _top_right_to_bottom_left_key(xy: np.ndarray) -> Tuple[float, float]:
        """Sort key: upper rows first (smaller y), then right to left (larger x first)."""
        x1, y1, x2, y2 = float(xy[0]), float(xy[1]), float(xy[2]), float(xy[3])
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        return (cy, -cx)

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
                    "lost_at": frame_idx,
                })
            self._prev = []
            self._lost = [x for x in self._lost if x["deadline"] > frame_idx]
            return np.array([], dtype=np.int32)

        xyxy = d.xyxy
        classes = d.class_id.astype(np.int32)
        hists = [self._bbox_hist(frame, xyxy[i]) for i in range(n)]

        stable = np.full(n, -1, dtype=np.int32)
        matched_prev_idx: set[int] = set()
        self._match_prev_iou_hungarian(
            n, xyxy, classes, stable, matched_prev_idx,
        )
        self._match_prev_center_fallback(
            n, xyxy, classes, stable, matched_prev_idx,
        )

        for pi, p in enumerate(self._prev):
            if pi not in matched_prev_idx:
                self._lost.append({
                    "stable_id": p.stable_id,
                    "class_id": p.class_id,
                    "hist": p.hist,
                    "last_xyxy": p.xyxy.copy(),
                    "deadline": frame_idx + self.lost_ttl_frames,
                    "lost_at": frame_idx,
                })

        self._lost = [x for x in self._lost if x["deadline"] > frame_idx]

        match_thresh = (
            self.combined_match_thresh
            if self.use_position_score
            else self.hist_match_thresh
        )

        gap_candidates: List[int] = []
        need_new_by_class: Dict[int, List[int]] = {}
        for j in range(n):
            if stable[j] != -1:
                continue
            hj = hists[j]
            if hj is None:
                gap_candidates.append(j)
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
                rk = -1
                rs = -1.0
                for k, L in enumerate(self._lost):
                    if L["class_id"] != int(classes[j]):
                        continue
                    lh = L["hist"]
                    if lh is None:
                        continue
                    lost_at = int(L.get("lost_at", -10**9))
                    if frame_idx - lost_at > self.recent_reid_max_frames:
                        continue
                    dist = self._center_dist(L["last_xyxy"], xyxy[j])
                    if dist > self.recent_reid_max_center_dist_px:
                        continue
                    hist_s = max(0.0, float(cv2.compareHist(hj, lh, cv2.HISTCMP_CORREL)))
                    if hist_s < self.recent_reid_min_hist:
                        continue
                    if self.use_position_score:
                        pos_s = self._position_score(
                            L["last_xyxy"], xyxy[j], self.position_sigma_px,
                        )
                        s = self._hist_w * hist_s + self._pos_w * pos_s
                    else:
                        s = hist_s
                    if s > rs:
                        rs = s
                        rk = k
                if rk >= 0 and rs >= self.combined_match_thresh * 0.82:
                    L = self._lost.pop(rk)
                    stable[j] = int(L["stable_id"])
                else:
                    gap_candidates.append(j)

        self._fill_gap_position_matches(xyxy, classes, stable, frame_idx, gap_candidates)

        for j in gap_candidates:
            if int(stable[j]) != -1:
                continue
            cid = int(classes[j])
            need_new_by_class.setdefault(cid, []).append(j)

        for cid in sorted(need_new_by_class.keys()):
            for j in sorted(
                need_new_by_class[cid],
                key=lambda jj: self._top_right_to_bottom_left_key(xyxy[jj]),
            ):
                stable[j] = self._new_id(cid)

        new_prev: List[_TrackState] = []
        for j in range(n):
            sid = int(stable[j])
            cid = int(classes[j])
            old_hist = None
            for p in self._prev:
                if p.stable_id == sid and p.class_id == cid:
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


def _roi_inclusive_bounds(
    frame_w: int, frame_h: int, roi_half_width: Optional[int],
) -> Optional[Tuple[int, int, int, int]]:
    """Return inclusive (x1, y1, x2, y2) in full-frame coords, or None = full frame."""
    if roi_half_width is None or roi_half_width <= 0:
        return None
    cx = frame_w // 2
    x1 = max(0, cx - roi_half_width)
    x2 = min(frame_w - 1, cx + roi_half_width)
    if x2 <= x1 + 4:
        return None
    y1, y2 = 0, frame_h - 1
    return (x1, y1, x2, y2)


def run(*, source="0", model_path=DEFAULT_MODEL, conf=0.8, nms=0.3,
        compensation=False, min_frames=1, stable_ids=True,
        lost_ttl_frames=900, byte_lost_buffer=120, hist_match_thresh=0.62,
        use_position_reid: bool = True, position_sigma_px: float = 200.0,
        combined_hist_weight: float = 0.65, combined_match_thresh: float = 0.55,
        output_path: Optional[str] = None, show_window: bool = True,
        roi_half_width: Optional[int] = None,
        count_roi_entries: bool = True,
        roi_count_use_stable_id: bool = False):
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
        output_path: If set, save annotated frames to this video file (e.g. .mp4).
        show_window: If False, skip imshow (use with output_path for batch encode).
        roi_half_width: If set, detect only on a full-height strip centered on the frame,
            half-width n px each side (inclusive span ~2n+1). None = full frame.
        count_roi_entries: If True and ROI is active, increment class counts when a
            track first enters the strip from the left outside (x < x1) or right
            outside (x > x2); each (class, id) counts at most once until reset.
        roi_count_use_stable_id: If False (default), ROI count keys use ByteTrack IDs
            (new track after dropout = new count). Safer when similar tomatoes occupy
            the same slot and stable re-ID would merge them. If True, count keys use
            stable display IDs (legacy; can under-count after false re-ID merge).
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
    if id_assigner is not None:
        print(id_assigner.config_log_line())
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
    roi_bounds = _roi_inclusive_bounds(w, h, roi_half_width)
    do_roi = roi_bounds is not None
    do_count = bool(do_roi and count_roi_entries)
    if roi_half_width is not None and roi_half_width > 0 and not do_roi:
        print("[WARN] ROI half-width too large or narrow for frame; using full frame.")
    if do_roi:
        rx1, ry1, rx2, ry2 = roi_bounds  # type: ignore[misc]
        print(
            f"[INFO] ROI strip (detect): x=[{rx1},{rx2}] y=[{ry1},{ry2}] "
            f"(center±{roi_half_width}px)",
        )
        if do_count:
            print(
                "[INFO] Entry count: 좌/우 바깥 → ROI 안; (class, id)당 1회까지 누적.",
            )
            if roi_count_use_stable_id:
                print(
                    "[INFO] ROI count id: stable 표시 번호 "
                    "(재식별이 합치면 카운트가 어긋날 수 있음).",
                )
            elif stable_ids:
                print(
                    "[INFO] ROI count id: ByteTrack | 라벨 번호: stable (표시만 재식별).",
                )
            else:
                print("[INFO] ROI count id: ByteTrack (stable 끔).")
        elif not count_roi_entries:
            print("[INFO] ROI entry counting OFF (--no-roi-count).")

    writer: Optional[cv2.VideoWriter] = None
    if output_path:
        outp = Path(output_path)
        outp.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(outp), fourcc, fps_hint, (w, h))
        if not writer.isOpened():
            raise RuntimeError(f"Cannot open VideoWriter for: {outp}")
        print(f"[INFO] Saving annotated video -> {outp.resolve()}")

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
    if do_count and not roi_count_use_stable_id and not stable_ids:
        print(
            "[WARN] ROI count uses ByteTrack id; 짧게 트랙이 끊기면 같은 과일이 두 번 셀 수 있음.",
        )
    if do_count and roi_count_use_stable_id and not stable_ids:
        print(
            "[WARN] --roi-count-stable-id 는 stable id가 꺼져 있으면 적용되지 않음; "
            "카운트는 ByteTrack id를 씁니다.",
        )
    if show_window:
        print("[INFO] Press 'q' to quit, 'r' to reset tracker, 'c' to toggle compensation")

    fps_avg = 0.0
    alpha = 0.1
    frame_idx = 0
    count_ripe = 0
    count_unripe = 0
    counted_keys: set[Tuple[int, int]] = set()
    arm_left: Dict[Tuple[int, int], bool] = {}
    arm_right: Dict[Tuple[int, int], bool] = {}

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

        if do_roi:
            rx1, ry1, rx2, ry2 = roi_bounds  # type: ignore[misc]
            roi_img = frame[ry1 : ry2 + 1, rx1 : rx2 + 1]
            results = model(roi_img, conf=conf, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results).with_nms(
                threshold=nms,
            )
            if len(detections):
                d_xy = np.asarray(detections.xyxy, dtype=np.float64).copy()
                d_xy[:, [0, 2]] += float(rx1)
                d_xy[:, [1, 3]] += float(ry1)
                detections.xyxy = d_xy
        else:
            results = model(frame, conf=conf, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results).with_nms(
                threshold=nms,
            )
        detections = tracker.update(detections)

        # ripe(0), unripe(1)만 남기기
        valid = [i for i, c in enumerate(detections.class_id)
                 if c in CLASS_NAMES] if detections.class_id is not None else []
        detections = detections[valid]

        byte_ids_for_count: Optional[np.ndarray] = None
        if detections.tracker_id is not None:
            byte_ids_for_count = detections.tracker_id.astype(np.int32, copy=True)

        if id_assigner is not None:
            stable = id_assigner.assign(frame_idx, frame, detections)
            detections.tracker_id = stable

        n_ripe = int((detections.class_id == 0).sum()) if detections.class_id is not None else 0
        n_unripe = int((detections.class_id == 1).sum()) if detections.class_id is not None else 0

        if roi_count_use_stable_id and stable_ids:
            count_tracker_ids = detections.tracker_id
        else:
            count_tracker_ids = byte_ids_for_count

        if do_count and len(detections) and count_tracker_ids is not None:
            rx1, _, rx2, _ = roi_bounds  # type: ignore[misc]
            for i in range(len(detections)):
                sid = int(count_tracker_ids[i])
                if sid < 0:
                    continue
                cls = int(detections.class_id[i])
                if cls not in CLASS_NAMES:
                    continue
                ck = (cls, sid)
                if ck in counted_keys:
                    continue
                x1b, _, x2b, _ = detections.xyxy[i]
                cx = 0.5 * (float(x1b) + float(x2b))
                if cx < float(rx1):
                    arm_left[ck] = True
                elif cx > float(rx2):
                    arm_right[ck] = True
                elif float(rx1) <= cx <= float(rx2):
                    via = None
                    if arm_left.get(ck):
                        via = "left"
                    elif arm_right.get(ck):
                        via = "right"
                    if via is not None:
                        if cls == 0:
                            count_ripe += 1
                        else:
                            count_unripe += 1
                        counted_keys.add(ck)
                        log_line = (
                            f"[COUNT] {CLASS_NAMES[cls]} count_id={sid} "
                            f"via={via} edge | total ripe={count_ripe} unripe={count_unripe}"
                        )
                        if (
                            stable_ids
                            and not roi_count_use_stable_id
                            and detections.tracker_id is not None
                        ):
                            log_line += f" (label #{int(detections.tracker_id[i])})"
                        print(log_line)

        annotated = frame.copy()
        if do_roi:
            rx1, ry1, rx2, ry2 = roi_bounds  # type: ignore[misc]
            cv2.rectangle(
                annotated, (rx1, ry1), (rx2, ry2), (255, 200, 0), 2,
            )
            cv2.line(annotated, (rx1, ry1), (rx1, ry2), (0, 255, 255), 2)
            cv2.line(annotated, (rx2, ry1), (rx2, ry2), (255, 0, 255), 2)
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

        hud = (
            f"Frame ripe/unripe: {n_ripe}/{n_unripe} | "
            f"FPS: {fps_avg:.1f}"
        )
        if do_count:
            hud = (
                f"Cumulative count ripe/unripe: {count_ripe}/{count_unripe} | "
                f"in-ROI now: {n_ripe}/{n_unripe} | FPS: {fps_avg:.1f}"
            )
        if stable_ids:
            hud = f"{hud} | {STABLE_ID_LOGIC_TAG}"
        cv2.putText(
            annotated,
            hud,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
        )

        if writer is not None:
            writer.write(annotated)

        if show_window:
            cv2.imshow("YOLO26 Real-time Tracking", annotated)

        frame_idx += 1

        key = (cv2.waitKey(1) & 0xFF) if show_window else 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            tracker.reset()
            if id_assigner is not None:
                id_assigner.reset()
            if motion_estimator is not None:
                motion_estimator.reset()
            counted_keys.clear()
            arm_left.clear()
            arm_right.clear()
            count_ripe = 0
            count_unripe = 0
            frame_idx = 0
            print("[INFO] Tracker reset (counts and entry arms cleared)")
        elif key == ord("c"):
            if motion_estimator is None:
                motion_estimator, motion_trace_ann = build_motion_compensator()
                print("[INFO] Compensation ON")
            else:
                motion_estimator = None
                motion_trace_ann = None
                print("[INFO] Compensation OFF")

    cap.release()
    if writer is not None:
        writer.release()
        print("[INFO] Output video write complete.")
    if show_window:
        cv2.destroyAllWindows()
    if do_count:
        print(
            f"[INFO] Session entry counts: ripe={count_ripe}, unripe={count_unripe}",
        )
    print("[INFO] Done.")


if __name__ == "__main__":
    run()
