"""
토마토 실시간 트래킹
YOLO + sv.ByteTrack (클래스별 독립) + Stable ID + ReID + Motion Compensation + Order Constraint

핵심 구조:
  basic_bytetracker.py 의 안정적인 기반 (클래스별 sv.ByteTrack, minimum_matching_threshold=0.8)
  위에 아래 기능을 추가:
    - ROI 기반 검출 및 Stable ID 발급
    - ReID: HSV 히스토그램 기반 색상 특징으로 가려짐 복구
    - Motion Compensation: 레일 카메라 이동 보정
    - Order Constraint: 군집 내 x좌표 순서 보존
    - 방향 자동 감지 (L2R / R2L, EMA 안정화)
    - 실시간 카운팅 (입구 라인 통과 기준)

사용법:
  scripts/realtime_tracking_new.py 의 CONFIG 수정 후 실행
"""

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import supervision as sv
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO
from trackers import MotionAwareTraceAnnotator, MotionEstimator

# CONFIG는 scripts/realtime_tracking_new.py 에서 주입
CONFIG: Dict = {}

CLASS_NAMES = {0: "ripe", 1: "unripe"}


# ---------------------------------------------------------------------------
# TrackState
# ---------------------------------------------------------------------------

@dataclass
class TrackState:
    stable_id: int
    xyxy: np.ndarray
    class_id: int
    tracker_id: Optional[int] = None
    reid_feature: Optional[np.ndarray] = None
    lost_frames: int = 0
    last_seen_frame: int = 0
    crossed_line: bool = False
    warped_xyxy: Optional[np.ndarray] = None
    counted: bool = False
    consecutive_frames: int = 0


# ---------------------------------------------------------------------------
# StableIdAssigner
# ---------------------------------------------------------------------------

class StableIdAssigner:
    """Stable ID 관리.

    ROI 내부 객체에만 ID를 발급한다 (외부는 -1).
    한번 발급된 ID는 재사용하지 않는다.

    매칭 순서:
      1단계: 같은 클래스의 ByteTrack tracker_id가 같으면 stable_id 유지
             (클래스별 분리 sv.ByteTrack 사용으로 tracker_id 안정성 보장)
      2단계: 헝가리안 + 중심 거리 + ReID + 구조 제약 (y축, 면적, 방향)
      3단계: 출구 쪽 신규 객체는 lost track 복구 시도 → 실패 시 새 ID
    """

    def __init__(self, debug: bool = False):
        self.debug = debug or CONFIG.get("debug", False)
        self._next_id: Dict[int, int] = {}
        self.active_tracks: List[TrackState] = []
        self.lost_tracks: List[TrackState] = []

        self.detected_direction: str = "UNKNOWN"
        self.prev_centers: Dict[int, float] = {}
        self.direction_ema: float = 0.0
        self._last_transform: Optional[np.ndarray] = None

    def reset(self):
        self._next_id.clear()
        self.active_tracks = []
        self.lost_tracks = []
        self.detected_direction = "UNKNOWN"
        self.prev_centers.clear()
        self.direction_ema = 0.0
        self._last_transform = None

    # ------------------------------------------------------------------
    # 내부 유틸
    # ------------------------------------------------------------------

    def _new_id(self, class_id: int) -> int:
        n = self._next_id.get(class_id, 1)
        self._next_id[class_id] = n + 1
        return n

    @staticmethod
    def _center(box: np.ndarray) -> Tuple[float, float]:
        return 0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])

    @staticmethod
    def _center_dist(a: np.ndarray, b: np.ndarray) -> float:
        ax, ay = 0.5 * (a[0] + a[2]), 0.5 * (a[1] + a[3])
        bx, by = 0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3])
        return float(np.hypot(ax - bx, ay - by))

    @staticmethod
    def _area(box: np.ndarray) -> float:
        return float((box[2] - box[0]) * (box[3] - box[1]))

    # ------------------------------------------------------------------
    # ReID
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_reid_feature(frame: np.ndarray, box: np.ndarray) -> np.ndarray:
        """HSV 히스토그램 + RGB 평균으로 ReID 특징 추출"""
        x1, y1, x2, y2 = box.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        bins = CONFIG.get("reid_hist_bins", 32)
        if x2 <= x1 or y2 <= y1:
            return np.zeros(bins * 3 + 3, dtype=np.float32)
        crop = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h = cv2.normalize(cv2.calcHist([hsv], [0], None, [bins], [0, 180]), None).flatten()
        s = cv2.normalize(cv2.calcHist([hsv], [1], None, [bins], [0, 256]), None).flatten()
        v = cv2.normalize(cv2.calcHist([hsv], [2], None, [bins], [0, 256]), None).flatten()
        rgb_mean = crop.mean(axis=(0, 1)) / 255.0
        return np.concatenate([h, s, v, rgb_mean]).astype(np.float32)

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    # ------------------------------------------------------------------
    # Motion compensation
    # ------------------------------------------------------------------

    @staticmethod
    def _warp_box(box: np.ndarray, transform: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = box
        corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float64)
        ones = np.ones((4, 1), dtype=np.float64)
        pts = np.hstack([corners, ones])
        if transform.shape == (2, 3):
            warped = (transform @ pts.T).T
        else:
            wh = (transform @ pts.T).T
            warped = wh[:, :2] / wh[:, 2:3]
        return np.array([warped[:, 0].min(), warped[:, 1].min(),
                         warped[:, 0].max(), warped[:, 1].max()], dtype=np.float32)

    def _warp_all_tracks(self, transform: Optional[np.ndarray]):
        valid = transform is not None and isinstance(transform, np.ndarray)
        for t in self.active_tracks + self.lost_tracks:
            t.warped_xyxy = self._warp_box(t.xyxy, transform) if valid else t.xyxy.copy()

    def _match_box(self, t: TrackState) -> np.ndarray:
        return t.warped_xyxy if t.warped_xyxy is not None else t.xyxy

    # ------------------------------------------------------------------
    # 구조 제약
    # ------------------------------------------------------------------

    def _structural_ok(self, prev_box: np.ndarray, curr_box: np.ndarray) -> bool:
        """y축, 면적, 방향 제약 검사"""
        pcx, pcy = self._center(prev_box)
        ccx, ccy = self._center(curr_box)

        if abs(ccy - pcy) > CONFIG.get("max_y_diff", 100.0):
            return False

        pa, ca = self._area(prev_box), self._area(curr_box)
        if pa > 0 and ca > 0:
            if max(pa, ca) / min(pa, ca) > CONFIG.get("max_area_ratio", 3.0):
                return False

        direction = self.detected_direction
        max_back = CONFIG.get("max_backward_x", 50.0)
        if direction == "L2R" and ccx < pcx - max_back:
            return False
        if direction == "R2L" and ccx > pcx + max_back:
            return False
        if direction == "UNKNOWN":
            if abs(ccx - pcx) > CONFIG.get("max_movement_unknown", 300):
                return False

        return True

    # ------------------------------------------------------------------
    # Order Constraint (LIS 기반)
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_order_constraint(
        matches: List[Tuple[int, int]],
        all_prev: List[TrackState],
        curr_xyxy: np.ndarray,
        prev_indices: List[int],
        det_indices: List[int],
    ) -> List[Tuple[int, int]]:
        if len(matches) < 2:
            return matches

        data = []
        for ii, jj in matches:
            p = all_prev[prev_indices[ii]]
            pb = p.warped_xyxy if p.warped_xyxy is not None else p.xyxy
            pcx = 0.5 * (pb[0] + pb[2])
            ccx = 0.5 * (curr_xyxy[det_indices[jj]][0] + curr_xyxy[det_indices[jj]][2])
            data.append((pcx, ccx, (ii, jj)))

        data.sort(key=lambda x: x[0])
        curr_xs = [d[1] for d in data]

        if all(curr_xs[i] <= curr_xs[i + 1] for i in range(len(curr_xs) - 1)):
            return matches

        # LIS (patience sort)
        n = len(curr_xs)
        tails, lis_pos = [], []
        for val in curr_xs:
            lo, hi = 0, len(tails)
            while lo < hi:
                mid = (lo + hi) // 2
                if tails[mid] <= val:
                    lo = mid + 1
                else:
                    hi = mid
            if lo == len(tails):
                tails.append(val)
            else:
                tails[lo] = val
            lis_pos.append(lo)

        lis_len = len(tails)
        members: Set[int] = set()
        k = lis_len - 1
        for i in range(n - 1, -1, -1):
            if lis_pos[i] == k:
                members.add(i)
                k -= 1
                if k < 0:
                    break

        return [data[i][2] for i in range(n) if i in members]

    # ------------------------------------------------------------------
    # 방향 감지
    # ------------------------------------------------------------------

    def _update_direction(self):
        camera_dx = 0.0
        if CONFIG.get("motion_compensation", False) and self._last_transform is not None:
            t = self._last_transform
            if t.shape[0] >= 2:
                camera_dx = float(t[0, 2])

        dxs = []
        for t in self.active_tracks:
            if t.stable_id in self.prev_centers:
                cx, _ = self._center(t.xyxy)
                dxs.append((cx - self.prev_centers[t.stable_id]) - camera_dx)

        if len(dxs) >= CONFIG.get("direction_min_tracks", 3):
            mean_dx = float(np.mean(dxs))
            alpha = CONFIG.get("direction_ema_alpha", 0.3)
            self.direction_ema = mean_dx if self.direction_ema == 0.0 else (
                alpha * mean_dx + (1 - alpha) * self.direction_ema
            )

            thr = CONFIG.get("direction_dx_threshold", 5.0)
            hys = CONFIG.get("direction_hysteresis", 2.0)
            prev = self.detected_direction

            if prev == "L2R":
                new = "R2L" if self.direction_ema < -(thr + hys) else (
                    "UNKNOWN" if self.direction_ema < -thr else "L2R"
                )
            elif prev == "R2L":
                new = "L2R" if self.direction_ema > (thr + hys) else (
                    "UNKNOWN" if self.direction_ema > thr else "R2L"
                )
            else:
                new = ("L2R" if self.direction_ema > thr else
                       "R2L" if self.direction_ema < -thr else "UNKNOWN")

            if self.debug and new != prev and new != "UNKNOWN":
                print(f"[DIR] {prev} → {new} (ema={self.direction_ema:.1f}px, n={len(dxs)})")
            self.detected_direction = new

        self.prev_centers = {
            t.stable_id: self._center(t.xyxy)[0] for t in self.active_tracks
        }

    def _on_entry_side(self, cx: float, roi: Tuple[int, int, int, int]) -> bool:
        mid = (roi[0] + roi[2]) / 2.0
        if self.detected_direction == "L2R":
            return cx < mid
        if self.detected_direction == "R2L":
            return cx > mid
        return True

    # ------------------------------------------------------------------
    # assign (메인 메서드)
    # ------------------------------------------------------------------

    def assign(
        self,
        frame_idx: int,
        frame: np.ndarray,
        dets: sv.Detections,
        roi: Optional[Tuple[int, int, int, int]] = None,
        tracker_ids: Optional[np.ndarray] = None,
        coord_transform: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """stable_id 배열 반환. ROI 외부 및 미할당 객체는 -1."""
        n = len(dets)
        if n == 0:
            self._age_lost(frame_idx)
            return np.array([], dtype=np.int32)

        if tracker_ids is None and dets.tracker_id is not None:
            tracker_ids = dets.tracker_id

        xyxy = dets.xyxy
        classes = dets.class_id.astype(np.int32)
        confs = dets.confidence
        stable = np.full(n, -1, dtype=np.int32)

        # ReID 특징 추출
        reid_features = None
        if CONFIG.get("use_reid", False):
            reid_features = np.array([
                self._extract_reid_feature(frame, box) for box in xyxy
            ])

        # Motion compensation: 이전 트랙 좌표 워핑
        valid_tf = coord_transform is not None and isinstance(coord_transform, np.ndarray)
        if CONFIG.get("motion_compensation", False) and valid_tf:
            self._warp_all_tracks(coord_transform)
            self._last_transform = coord_transform
            if self.debug and frame_idx % 100 == 0:
                dx = coord_transform[0, 2] if coord_transform.shape[0] >= 2 else 0
                dy = coord_transform[1, 2] if coord_transform.shape[0] >= 2 else 0
                print(f"[MOTION] frame={frame_idx} Δx={dx:.1f} Δy={dy:.1f}")
        else:
            self._warp_all_tracks(None)

        # ROI 마스크
        if roi is not None:
            cx_arr = 0.5 * (xyxy[:, 0] + xyxy[:, 2])
            in_roi = (cx_arr >= roi[0]) & (cx_arr <= roi[2])
        else:
            in_roi = np.ones(n, dtype=bool)

        roi_idx = [j for j in range(n) if in_roi[j]]
        all_prev = self.active_tracks + self.lost_tracks
        used_prev: Set[int] = set()

        # ---------------------------------------------------------------
        # 1단계: ByteTrack tracker_id 기반 매칭
        #   클래스별 sv.ByteTrack 덕분에 tracker_id가 클래스 내에서 유일하고 안정적
        # ---------------------------------------------------------------
        if tracker_ids is not None and len(tracker_ids) == n:
            for j in roi_idx:
                tid = int(tracker_ids[j])
                if tid < 0:
                    continue

                # active tracks 우선
                for pi, p in enumerate(self.active_tracks):
                    if pi in used_prev:
                        continue
                    if int(p.class_id) != int(classes[j]):
                        continue
                    if p.tracker_id is None or int(p.tracker_id) != tid:
                        continue
                    if not self._structural_ok(self._match_box(p), xyxy[j]):
                        continue
                    stable[j] = p.stable_id
                    used_prev.add(pi)
                    break

                # active에서 못 찾으면 lost에서 시도 (거리 조건 추가)
                if stable[j] == -1:
                    for pi, p in enumerate(self.lost_tracks):
                        api = pi + len(self.active_tracks)
                        if api in used_prev:
                            continue
                        if int(p.class_id) != int(classes[j]):
                            continue
                        if p.tracker_id is None or int(p.tracker_id) != tid:
                            continue
                        if self._center_dist(self._match_box(p), xyxy[j]) > 150:
                            continue
                        if not self._structural_ok(self._match_box(p), xyxy[j]):
                            continue
                        stable[j] = p.stable_id
                        used_prev.add(api)
                        break

        # ---------------------------------------------------------------
        # 2단계: 헝가리안 매칭 (중심 거리 + ReID + 구조 제약)
        # ---------------------------------------------------------------
        for cid in {int(classes[j]) for j in roi_idx}:
            prev_idx = [i for i, p in enumerate(all_prev)
                        if p.class_id == cid and i not in used_prev]
            det_idx  = [j for j in roi_idx
                        if classes[j] == cid and stable[j] == -1]
            if not prev_idx or not det_idx:
                continue

            cost = np.full((len(prev_idx), len(det_idx)), 9999.0)
            use_reid = CONFIG.get("use_reid", False)
            reid_w   = CONFIG.get("reid_weight", 0.3)
            max_dist = CONFIG.get("center_max_dist", 200)

            for ii, pi in enumerate(prev_idx):
                p = all_prev[pi]
                pb = self._match_box(p)
                for jj, dj in enumerate(det_idx):
                    if not self._structural_ok(pb, xyxy[dj]):
                        continue
                    dist = self._center_dist(pb, xyxy[dj])
                    if use_reid and reid_features is not None:
                        dn = min(dist / max_dist, 1.0)
                        sim = self._cosine_sim(p.reid_feature, reid_features[dj]) if p.reid_feature is not None else 0.0
                        cost[ii, jj] = (1 - reid_w) * dn + reid_w * (1 - sim)
                    else:
                        cost[ii, jj] = dist

            ri, ci = linear_sum_assignment(cost)
            raw = list(zip(ri.tolist(), ci.tolist()))

            if CONFIG.get("use_order_constraint", False) and len(raw) >= 2:
                valid = self._apply_order_constraint(raw, all_prev, xyxy, prev_idx, det_idx)
                if self.debug and len(valid) < len(raw):
                    print(f"[ORDER] {len(raw)-len(valid)} match(es) rejected (class={cid})")
            else:
                valid = raw

            for ii, jj in valid:
                pi, dj = prev_idx[ii], det_idx[jj]
                p = all_prev[pi]
                dist = self._center_dist(self._match_box(p), xyxy[dj])

                if use_reid and reid_features is not None:
                    # 거리 기반 적응적 ReID 임계값
                    if dist <= 50:
                        stable[dj] = p.stable_id
                        used_prev.add(pi)
                    elif dist <= 200:
                        sim = self._cosine_sim(p.reid_feature, reid_features[dj]) if p.reid_feature is not None else 0.0
                        if sim >= CONFIG.get("reid_threshold", 0.5):
                            stable[dj] = p.stable_id
                            used_prev.add(pi)
                            if self.debug:
                                cx, cy = self._center(xyxy[dj])
                                print(f"[MID] {CLASS_NAMES.get(cid)} #{p.stable_id} ({cx:.0f},{cy:.0f}) dist={dist:.1f} sim={sim:.2f}")
                    elif dist <= 300:
                        sim = self._cosine_sim(p.reid_feature, reid_features[dj]) if p.reid_feature is not None else 0.0
                        if sim >= 0.7:
                            stable[dj] = p.stable_id
                            used_prev.add(pi)
                            if self.debug:
                                cx, cy = self._center(xyxy[dj])
                                print(f"[FAR] {CLASS_NAMES.get(cid)} #{p.stable_id} ({cx:.0f},{cy:.0f}) dist={dist:.1f} sim={sim:.2f}")
                else:
                    if dist <= max_dist:
                        stable[dj] = p.stable_id
                        used_prev.add(pi)

        # ---------------------------------------------------------------
        # 3단계: 신규 진입 처리 (출구 쪽은 lost 복구 시도 후 신규 ID)
        # 정렬 기준: (class_id, cy - cx) 오름차순
        #   cy - cx 가 가장 작은 쪽 = 우상단 → ID 작음
        #   cy - cx 가 가장 큰  쪽 = 좌하단 → ID 큼
        # ---------------------------------------------------------------
        new_entries = [(j, int(classes[j])) for j in roi_idx if stable[j] == -1]
        new_entries.sort(key=lambda item: (
            item[1],
            self._center(xyxy[item[0]])[1] - self._center(xyxy[item[0]])[0],  # cy - cx
        ))
        used_stable: Set[int] = {int(stable[j]) for j in roi_idx if stable[j] != -1}

        for j, cid in new_entries:
            cx, cy = self._center(xyxy[j])
            suspicious = (
                roi is not None
                and self.detected_direction != "UNKNOWN"
                and not self._on_entry_side(cx, roi)
            )
            if suspicious and self.debug:
                print(f"[SUSPICIOUS] {CLASS_NAMES.get(cid)} x={cx:.0f} (exit side, dir={self.detected_direction})")

            recovered = False
            if suspicious:
                best, best_cost = None, 1e9
                max_rec = CONFIG.get("suspicious_new_match_dist", 350)
                reid_thr = CONFIG.get("suspicious_recover_reid_threshold", 0.35)
                lf_pen = CONFIG.get("suspicious_lost_frames_penalty", 2.0)

                for p in self.lost_tracks:
                    if p.stable_id in used_stable or p.class_id != cid:
                        continue
                    if not self._structural_ok(self._match_box(p), xyxy[j]):
                        continue
                    dist = self._center_dist(self._match_box(p), xyxy[j])
                    if dist > max_rec:
                        continue
                    c = dist
                    if CONFIG.get("use_reid", False) and reid_features is not None and p.reid_feature is not None:
                        sim = self._cosine_sim(p.reid_feature, reid_features[j])
                        if sim < reid_thr:
                            continue
                        c += (1 - sim) * 100 * 0.5
                    c += p.lost_frames * lf_pen
                    if c < best_cost:
                        best_cost, best = c, p

                if best is not None:
                    stable[j] = best.stable_id
                    used_stable.add(best.stable_id)
                    recovered = True
                    if self.debug:
                        dist = self._center_dist(self._match_box(best), xyxy[j])
                        print(f"[RECOVER] {CLASS_NAMES.get(cid)} #{best.stable_id} ({cx:.0f},{cy:.0f}) dist={dist:.1f} cost={best_cost:.1f}")

            if not recovered:
                new_id = self._new_id(cid)
                stable[j] = new_id
                used_stable.add(new_id)
                if self.debug:
                    sfx = " ⚠️" if suspicious else ""
                    print(f"[NEW] {CLASS_NAMES.get(cid)} #{new_id} ({cx:.0f},{cy:.0f}){sfx}")

        # 트랙 상태 업데이트
        self._update_tracks(frame_idx, roi_idx, stable, xyxy, classes, tracker_ids, reid_features, n)
        return stable

    # ------------------------------------------------------------------
    # 트랙 상태 관리
    # ------------------------------------------------------------------

    def _update_tracks(
        self,
        frame_idx: int,
        roi_idx: List[int],
        stable: np.ndarray,
        xyxy: np.ndarray,
        classes: np.ndarray,
        tracker_ids: Optional[np.ndarray],
        reid_features: Optional[np.ndarray],
        n: int,
    ):
        matched_ids = {int(stable[j]) for j in roi_idx if stable[j] != -1}
        prev_state: Dict[Tuple[int, int], Tuple[bool, int]] = {
            (t.class_id, t.stable_id): (t.counted, t.consecutive_frames)
            for t in self.active_tracks + self.lost_tracks
        }

        new_active = []
        for j in roi_idx:
            if stable[j] == -1:
                continue
            tid = None
            if tracker_ids is not None and len(tracker_ids) == n:
                v = int(tracker_ids[j])
                if v >= 0:
                    tid = v
            sid = int(stable[j])
            cid = int(classes[j])
            prev_counted, prev_consec = prev_state.get((cid, sid), (False, 0))
            new_active.append(TrackState(
                stable_id=sid,
                xyxy=xyxy[j].copy(),
                class_id=cid,
                tracker_id=tid,
                reid_feature=reid_features[j] if reid_features is not None else None,
                lost_frames=0,
                last_seen_frame=frame_idx,
                counted=prev_counted,
                consecutive_frames=prev_consec + 1,
            ))

        buf_short = CONFIG.get("lost_buffer_frames", 20)
        buf_long  = CONFIG.get("lost_buffer_uncounted", 150)

        new_lost = []
        for t in self.active_tracks:
            if t.stable_id not in matched_ids:
                t.lost_frames += 1
                t.consecutive_frames = 0
                lim = buf_long if not t.counted else buf_short
                if t.lost_frames <= lim:
                    new_lost.append(t)
        for t in self.lost_tracks:
            if t.stable_id not in matched_ids:
                t.lost_frames += 1
                lim = buf_long if not t.counted else buf_short
                if t.lost_frames <= lim:
                    new_lost.append(t)

        self.active_tracks = new_active
        self.lost_tracks   = new_lost
        self._update_direction()

    def _age_lost(self, frame_idx: int):
        """검출이 없는 프레임에서 lost 트랙 나이 증가"""
        buf_short = CONFIG.get("lost_buffer_frames", 20)
        buf_long  = CONFIG.get("lost_buffer_uncounted", 150)
        new_lost = []
        for t in self.active_tracks:
            t.lost_frames += 1
            t.consecutive_frames = 0
            if t.lost_frames <= (buf_long if not t.counted else buf_short):
                new_lost.append(t)
        for t in self.lost_tracks:
            t.lost_frames += 1
            if t.lost_frames <= (buf_long if not t.counted else buf_short):
                new_lost.append(t)
        self.active_tracks = []
        self.lost_tracks   = new_lost


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def run(
    source: str = "notebook/rgb.mp4",
    model_path: Optional[str] = None,
    roi_half_width: int = 320,
    output_path: Optional[str] = None,
    show_window: bool = True,
    save_results: Optional[str] = None,
):
    """
    토마토 트래킹 실행

    Args:
        source:         영상 소스 (0=웹캠, 파일 경로)
        model_path:     YOLO 모델 경로 (None=기본)
        roi_half_width: ROI 반폭 픽셀 (None=전체 프레임)
        output_path:    출력 영상 저장 경로
        show_window:    창 표시 여부
        save_results:   CSV/JSON 저장 기본 경로
    """
    if model_path is None:
        model_path = str(
            Path(__file__).parent.parent / "runs" / "yolo26_custom_tomato" / "trained_yolo26_custom.pt"
        )

    model = YOLO(model_path)
    cap   = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {source}")

    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # ROI
    roi = None
    if roi_half_width:
        cx = w // 2
        roi = (max(0, cx - roi_half_width), 0, min(w - 1, cx + roi_half_width), h - 1)
        print(f"[INFO] ROI: x=[{roi[0]}, {roi[2]}]")

    # ── 클래스별 독립 sv.ByteTrack (basic_bytetracker.py 와 동일 구조) ──────
    def _make_bytetrack() -> sv.ByteTrack:
        return sv.ByteTrack(
            track_activation_threshold=CONFIG.get("byte_track_activation_threshold", 0.25),
            lost_track_buffer=CONFIG.get("byte_buffer", 30),
            minimum_matching_threshold=CONFIG.get("byte_minimum_matching_threshold", 0.8),
            frame_rate=fps,
        )

    trackers: Dict[int, sv.ByteTrack] = {cid: _make_bytetrack() for cid in CLASS_NAMES}
    id_assigner = StableIdAssigner()

    # ── Annotators ────────────────────────────────────────────────────────
    colors    = sv.ColorPalette.from_hex(["#FF0000", "#00CC00"])
    box_ann   = sv.BoxAnnotator(color=colors, color_lookup=sv.ColorLookup.CLASS)
    label_ann = sv.LabelAnnotator(
        color=colors, color_lookup=sv.ColorLookup.CLASS,
        text_color=sv.Color.WHITE, text_scale=0.42, text_thickness=1, text_padding=4,
    )
    trace_ann = sv.TraceAnnotator(
        color=colors, color_lookup=sv.ColorLookup.CLASS,
        thickness=2, trace_length=int(CONFIG.get("trace_length", 80)),
    )

    motion_estimator: Optional[MotionEstimator] = None
    motion_trace_ann: Optional[MotionAwareTraceAnnotator] = None
    if CONFIG.get("motion_compensation", False):
        motion_estimator = MotionEstimator(
            max_points=int(CONFIG.get("motion_max_points", 500)),
            min_distance=int(CONFIG.get("motion_min_distance", 10)),
            block_size=int(CONFIG.get("motion_block_size", 3)),
            quality_level=float(CONFIG.get("motion_quality_level", 0.001)),
            ransac_reproj_threshold=float(CONFIG.get("motion_ransac_reproj_threshold", 1.0)),
        )
        motion_trace_ann = MotionAwareTraceAnnotator(
            color=colors, color_lookup=sv.ColorLookup.CLASS,
            thickness=2, trace_length=int(CONFIG.get("trace_length", 80)),
        )

    writer = None
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    count_ripe, count_unripe = 0, 0
    frame_log: List[Tuple] = []
    prev_positions: Dict[Tuple[int, int], float] = {}
    counted_ids: Set[Tuple[int, int]] = set()

    print(f"[INFO] Source: {source} ({w}x{h} @ {fps:.1f}fps)")
    print(f"[INFO] Model: {model_path}")
    print(f"[INFO] ByteTrack: activation={CONFIG.get('byte_track_activation_threshold',0.25)}"
          f" matching={CONFIG.get('byte_minimum_matching_threshold',0.8)}"
          f" buffer={CONFIG.get('byte_buffer',30)}")
    print(f"[INFO] ReID: {'on' if CONFIG.get('use_reid') else 'off'}"
          f" | Motion: {'on' if CONFIG.get('motion_compensation') else 'off'}"
          f" | Order: {'on' if CONFIG.get('use_order_constraint') else 'off'}")

    fps_avg  = 0.0
    frame_idx = 0

    while True:
        t0 = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break

        coord_transform = None
        if motion_estimator is not None:
            coord_transform = motion_estimator.update(frame)

        # ── YOLO 검출 (ROI 크롭 or 전체 프레임) ──────────────────────────
        if roi is not None:
            x0, y0, x1, y1 = roi
            crop = frame[y0:y1 + 1, x0:x1 + 1]
            if crop.size == 0:
                all_dets = sv.Detections.empty()
            else:
                res = model(crop, conf=CONFIG.get("conf", 0.5), verbose=False)[0]
                all_dets = sv.Detections.from_ultralytics(res).with_nms(
                    threshold=CONFIG.get("nms", 0.3)
                )
                if len(all_dets) > 0:
                    off = np.array([x0, y0, x0, y0], dtype=np.float32)
                    all_dets.xyxy = all_dets.xyxy + off
        else:
            res = model(frame, conf=CONFIG.get("conf", 0.5), verbose=False)[0]
            all_dets = sv.Detections.from_ultralytics(res).with_nms(
                threshold=CONFIG.get("nms", 0.3)
            )

        # ── 클래스별 ByteTrack 업데이트 (basic_bytetracker.py 방식) ──────
        boxes_l, confs_l, cls_l, tid_l = [], [], [], []
        for cid, btracker in trackers.items():
            mask     = all_dets.class_id == cid
            cls_dets = all_dets[mask]
            if len(cls_dets) == 0:
                continue
            cls_dets = btracker.update_with_detections(cls_dets)
            if cls_dets.tracker_id is not None and len(cls_dets) > 0:
                boxes_l.append(cls_dets.xyxy)
                confs_l.append(cls_dets.confidence)
                cls_l.append(cls_dets.class_id)
                tid_l.append(cls_dets.tracker_id)

        if boxes_l:
            dets = sv.Detections(
                xyxy=np.concatenate(boxes_l),
                confidence=np.concatenate(confs_l),
                class_id=np.concatenate(cls_l),
                tracker_id=np.concatenate(tid_l),
            )
        else:
            dets = sv.Detections.empty()

        # 유효 클래스 필터
        if dets.class_id is not None and len(dets) > 0:
            valid = [i for i, c in enumerate(dets.class_id) if c in CLASS_NAMES]
            dets = dets[valid]

        # ByteTrack ID 보존 (궤적용)
        bytetrack_ids = dets.tracker_id.copy() if dets.tracker_id is not None else None

        # ── Stable ID 발급 ────────────────────────────────────────────────
        stable_ids = id_assigner.assign(
            frame_idx, frame, dets, roi=roi,
            tracker_ids=bytetrack_ids,
            coord_transform=coord_transform,
        )

        # ── 카운팅 ────────────────────────────────────────────────────────
        if roi and len(dets) > 0 and len(stable_ids) > 0:
            direction    = id_assigner.detected_direction
            entry_offset = CONFIG.get("counting_entry_offset", 50)
            min_consec   = CONFIG.get("counting_min_consecutive", 3)

            if direction == "L2R":
                line_x = roi[0] + entry_offset
            elif direction == "R2L":
                line_x = roi[2] - entry_offset
            else:
                line_x = (roi[0] + roi[2]) // 2

            for i in range(len(dets)):
                sid = int(stable_ids[i])
                if sid == -1:
                    continue
                cid = int(dets.class_id[i])
                key = (cid, sid)
                cx  = 0.5 * (dets.xyxy[i][0] + dets.xyxy[i][2])

                if key in prev_positions:
                    prev_cx = prev_positions[key]
                    crossed = False
                    if direction == "L2R":
                        crossed = prev_cx < line_x <= cx
                    elif direction == "R2L":
                        crossed = prev_cx > line_x >= cx
                    else:
                        ema_thr = CONFIG.get("count_unknown_ema_threshold", 3.0)
                        if abs(id_assigner.direction_ema) >= ema_thr:
                            if id_assigner.direction_ema > 0:
                                crossed = prev_cx < line_x <= cx
                            else:
                                crossed = prev_cx > line_x >= cx

                    consec = next(
                        (t.consecutive_frames for t in id_assigner.active_tracks
                         if t.class_id == cid and t.stable_id == sid), 0
                    )
                    if crossed and key not in counted_ids and consec >= min_consec:
                        if cid == 0:
                            count_ripe += 1
                        else:
                            count_unripe += 1
                        counted_ids.add(key)
                        for t in id_assigner.active_tracks:
                            if t.class_id == cid and t.stable_id == sid:
                                t.counted = True
                                break
                        if CONFIG.get("debug"):
                            print(f"[COUNT] {CLASS_NAMES.get(cid)} #{sid} x={line_x} ({direction}) consec={consec}")

                prev_positions[key] = cx

        # 사라진 트랙 위치 정리
        alive = {(t.class_id, t.stable_id)
                 for t in id_assigner.active_tracks + id_assigner.lost_tracks}
        prev_positions = {k: v for k, v in prev_positions.items() if k in alive}

        # ── 디버그 로그 ──────────────────────────────────────────────────
        if CONFIG.get("debug") and len(stable_ids) > 0:
            r_ids = sorted(int(stable_ids[i]) for i in range(len(dets))
                           if stable_ids[i] != -1 and dets.class_id[i] == 0)
            u_ids = sorted(int(stable_ids[i]) for i in range(len(dets))
                           if stable_ids[i] != -1 and dets.class_id[i] == 1)
            print(f"[FRAME {frame_idx:04d}] ripe={r_ids} unripe={u_ids}")

        # ── 시각화 ────────────────────────────────────────────────────────
        vis = frame.copy()

        if roi:
            cv2.rectangle(vis, (roi[0], roi[1]), (roi[2], roi[3]), (255, 200, 0), 2)
            direction = id_assigner.detected_direction
            entry_offset = CONFIG.get("counting_entry_offset", 50)
            if direction == "L2R":
                lx = roi[0] + entry_offset
            elif direction == "R2L":
                lx = roi[2] - entry_offset
            else:
                lx = (roi[0] + roi[2]) // 2
            cv2.line(vis, (lx, roi[1]), (lx, roi[3]), (0, 255, 255), 2)

        vis = box_ann.annotate(vis, dets)

        if CONFIG.get("show_trace", False) and bytetrack_ids is not None:
            dets.tracker_id = bytetrack_ids
            if motion_trace_ann is not None:
                vis = motion_trace_ann.annotate(vis, dets, coord_transform=coord_transform)
            else:
                vis = trace_ann.annotate(vis, dets)

        dets.tracker_id = stable_ids
        labels = []
        for i in range(len(dets)):
            cname = CLASS_NAMES.get(int(dets.class_id[i]), "?")
            sid   = int(dets.tracker_id[i])
            labels.append(cname if sid == -1 else f"{cname} #{sid}")
        vis = label_ann.annotate(vis, dets, labels)

        dt      = time.perf_counter() - t0
        fps_now = 1.0 / dt if dt > 0 else 0
        fps_avg = 0.1 * fps_now + 0.9 * fps_avg if fps_avg else fps_now

        n_r = int((dets.class_id == 0).sum()) if len(dets) else 0
        n_u = int((dets.class_id == 1).sum()) if len(dets) else 0
        hud = f"Count: {count_ripe}R/{count_unripe}U | Now: {n_r}R/{n_u}U | FPS: {fps_avg:.1f}"
        cv2.putText(vis, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        frame_log.append((frame_idx, n_r, n_u, count_ripe, count_unripe))

        if writer:
            writer.write(vis)

        if show_window:
            cv2.imshow("Tomato Tracker", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                for bt in trackers.values():
                    bt.reset()
                id_assigner.reset()
                if motion_estimator is not None:
                    motion_estimator.reset()
                count_ripe = count_unripe = 0
                prev_positions.clear()
                counted_ids.clear()
                frame_log.clear()
                frame_idx = 0
                print("[RESET]")

        frame_idx += 1

    cap.release()
    if writer:
        writer.release()
    if show_window:
        cv2.destroyAllWindows()

    print(f"[DONE] ripe={count_ripe}, unripe={count_unripe}")

    if save_results:
        base = Path(save_results)
        base.parent.mkdir(parents=True, exist_ok=True)
        with open(base.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
            w_ = csv.writer(f)
            w_.writerow(["frame", "n_ripe", "n_unripe", "total_ripe", "total_unripe"])
            w_.writerows(frame_log)
        with open(base.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump({"total_ripe": count_ripe, "total_unripe": count_unripe,
                       "total_frames": frame_idx}, f, indent=2)
        print(f"[SAVED] {base}.csv / .json")


# ---------------------------------------------------------------------------
# run_benchmark  (benchmark.py에서 import해서 사용)
# ---------------------------------------------------------------------------

def run_benchmark(config: dict) -> dict:
    """벤치마크용 트래킹 실행 후 benchmark 호환 결과 반환.

    benchmark.py 의 CONFIG 키를 그대로 받아서 내부 CONFIG를 구성합니다.

    Returns:
        dict:
            mot_rows    (List[tuple]): (frame_id, track_id, x, y, w, h, conf, class_id)
            fps_avg     (float)
            total_frames (int)
            unique_ids  (Dict[int, set]): {class_id: set of stable_ids}
    """
    import time as _time

    # benchmark CONFIG 키 → tracker 모듈 CONFIG 키 매핑
    global CONFIG
    CONFIG = {
        "conf":    config.get("conf", 0.5),
        "nms":     config.get("iou", 0.3),
        "byte_track_activation_threshold": config.get("byte_track_activation_threshold", 0.25),
        "byte_minimum_matching_threshold": config.get("byte_minimum_matching_threshold", 0.8),
        "byte_buffer":             config.get("byte_lost_track_buffer", 30),
        "max_y_diff":              config.get("tnew_max_y_diff", 100.0),
        "max_area_ratio":          config.get("tnew_max_area_ratio", 3.0),
        "max_backward_x":          config.get("tnew_max_backward_x", 50.0),
        "max_movement_unknown":    config.get("tnew_max_movement_unknown", 300),
        "center_max_dist":         config.get("tnew_center_max_dist", 200),
        "use_reid":                config.get("tnew_use_reid", True),
        "reid_weight":             config.get("tnew_reid_weight", 0.3),
        "reid_threshold":          config.get("tnew_reid_threshold", 0.5),
        "reid_hist_bins":          config.get("tnew_reid_hist_bins", 32),
        "lost_buffer_frames":      config.get("tnew_lost_buffer_frames", 20),
        "lost_buffer_uncounted":   config.get("tnew_lost_buffer_uncounted", 150),
        "use_order_constraint":    config.get("tnew_use_order_constraint", True),
        "motion_compensation":     config.get("tnew_motion_compensation", True),
        "motion_max_points":       config.get("tnew_motion_max_points", 500),
        "motion_min_distance":     config.get("tnew_motion_min_distance", 10),
        "motion_block_size":       config.get("tnew_motion_block_size", 3),
        "motion_quality_level":    config.get("tnew_motion_quality_level", 0.001),
        "motion_ransac_reproj_threshold": config.get("tnew_motion_ransac_reproj_threshold", 1.0),
        "direction_min_tracks":    config.get("tnew_direction_min_tracks", 3),
        "direction_dx_threshold":  config.get("tnew_direction_dx_threshold", 5.0),
        "direction_ema_alpha":     config.get("tnew_direction_ema_alpha", 0.3),
        "direction_hysteresis":    config.get("tnew_direction_hysteresis", 2.0),
        "suspicious_new_match_dist":          config.get("tnew_suspicious_new_match_dist", 350),
        "suspicious_recover_reid_threshold":  config.get("tnew_suspicious_recover_reid_threshold", 0.35),
        "suspicious_lost_frames_penalty":     config.get("tnew_suspicious_lost_frames_penalty", 2.0),
        "count_unknown_ema_threshold":        config.get("tnew_count_unknown_ema_threshold", 3.0),
        "counting_entry_offset":              config.get("tnew_counting_entry_offset", 50),
        "counting_min_consecutive":           config.get("tnew_counting_min_consecutive", 3),
        "trace_length":            config.get("tnew_trace_length", 80),
        "debug":                   False,
    }

    import supervision as sv
    from trackers import MotionEstimator

    source    = config.get("source", "notebook/rgb.mp4")
    model_path = config.get("model_path", None)
    roi_hw    = config.get("tnew_roi_half_width", 320)
    output_path = config.get("output_path")

    if model_path is None:
        model_path = str(
            Path(__file__).parent.parent / "runs" / "yolo26_custom_tomato" / "trained_yolo26_custom.pt"
        )

    model = YOLO(model_path)
    vid   = int(source) if str(source).isdigit() else source
    cap   = cv2.VideoCapture(vid)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {source}")

    w_   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_ = cap.get(cv2.CAP_PROP_FPS) or 30.0

    roi = None
    if roi_hw:
        cx_ = w_ // 2
        roi = (max(0, cx_ - roi_hw), 0, min(w_ - 1, cx_ + roi_hw), h_ - 1)

    def _make_bt():
        return sv.ByteTrack(
            track_activation_threshold=CONFIG.get("byte_track_activation_threshold", 0.25),
            lost_track_buffer=CONFIG.get("byte_buffer", 30),
            minimum_matching_threshold=CONFIG.get("byte_minimum_matching_threshold", 0.8),
            frame_rate=fps_,
        )

    trackers_bt  = {cid: _make_bt() for cid in CLASS_NAMES}
    id_assigner  = StableIdAssigner()
    motion_est: Optional[MotionEstimator] = None
    if CONFIG.get("motion_compensation", False):
        motion_est = MotionEstimator(
            max_points=int(CONFIG.get("motion_max_points", 500)),
            min_distance=int(CONFIG.get("motion_min_distance", 10)),
            block_size=int(CONFIG.get("motion_block_size", 3)),
            quality_level=float(CONFIG.get("motion_quality_level", 0.001)),
            ransac_reproj_threshold=float(CONFIG.get("motion_ransac_reproj_threshold", 1.0)),
        )

    writer_ = None
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        writer_ = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps_, (w_, h_))

    seen_ids: Dict[int, set] = {cid: set() for cid in CLASS_NAMES}
    mot_rows = []
    frame_idx_, fps_acc = 0, 0.0

    print(f"[tracker] 시작...")

    while True:
        t0 = _time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx_ += 1

        coord_tf = motion_est.update(frame) if motion_est else None

        if roi is not None:
            x0_, y0_, x1_, y1_ = roi
            crop = frame[y0_:y1_ + 1, x0_:x1_ + 1]
            if crop.size == 0:
                all_dets = sv.Detections.empty()
            else:
                res = model(crop, conf=CONFIG.get("conf", 0.5), verbose=False)[0]
                all_dets = sv.Detections.from_ultralytics(res).with_nms(
                    threshold=CONFIG.get("nms", 0.3)
                )
                if len(all_dets) > 0:
                    off = np.array([x0_, y0_, x0_, y0_], dtype=np.float32)
                    all_dets.xyxy = all_dets.xyxy + off
        else:
            res = model(frame, conf=CONFIG.get("conf", 0.5), verbose=False)[0]
            all_dets = sv.Detections.from_ultralytics(res).with_nms(
                threshold=CONFIG.get("nms", 0.3)
            )

        boxes_l, confs_l, cls_l, tid_l = [], [], [], []
        for cid, trk in trackers_bt.items():
            mask = all_dets.class_id == cid
            cd   = all_dets[mask]
            if len(cd) == 0:
                continue
            cd = trk.update_with_detections(cd)
            if cd.tracker_id is not None and len(cd) > 0:
                boxes_l.append(cd.xyxy)
                confs_l.append(cd.confidence)
                cls_l.append(cd.class_id)
                tid_l.append(cd.tracker_id)

        if boxes_l:
            dets = sv.Detections(
                xyxy=np.concatenate(boxes_l),
                confidence=np.concatenate(confs_l),
                class_id=np.concatenate(cls_l),
                tracker_id=np.concatenate(tid_l),
            )
        else:
            dets = sv.Detections.empty()

        bt_ids     = dets.tracker_id.copy() if dets.tracker_id is not None else None
        stable_ids = id_assigner.assign(
            frame_idx_, frame, dets, roi=roi,
            tracker_ids=bt_ids, coord_transform=coord_tf,
        )

        if len(dets) > 0 and len(stable_ids) > 0:
            for i in range(len(dets)):
                sid = int(stable_ids[i])
                if sid == -1:
                    continue
                cid  = int(dets.class_id[i])
                x1, y1, x2, y2 = dets.xyxy[i].astype(int)
                conf_v = float(dets.confidence[i]) if dets.confidence is not None else 0.0
                seen_ids[cid].add(sid)
                mot_rows.append((frame_idx_, sid, x1, y1, x2 - x1, y2 - y1, conf_v, cid))

                if writer_:
                    color = (60, 80, 255) if cid == 0 else (60, 200, 80)
                    label = f"{CLASS_NAMES.get(cid, cid)} #{sid} {conf_v:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 2, y1), color, -1)
                    cv2.putText(frame, label, (x1 + 1, y1 - 3),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if writer_:
            if roi:
                cv2.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]), (255, 200, 0), 2)
            for i, text in enumerate([
                f"[tracker] Frame {frame_idx_}",
                f"ripe   IDs: {len(seen_ids[0])}",
                f"unripe IDs: {len(seen_ids[1])}",
            ]):
                y = 30 + i * 28
                cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 3)
                cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1)
            writer_.write(frame)

        elapsed = _time.perf_counter() - t0
        fps_now  = 1.0 / elapsed if elapsed > 0 else 0
        fps_acc  = 0.1 * fps_now + 0.9 * fps_acc if fps_acc else fps_now

    cap.release()
    if writer_:
        writer_.release()

    print(f"[tracker] 완료 | {frame_idx_}프레임 | FPS={fps_acc:.1f} | "
          f"ripe={len(seen_ids[0])} unripe={len(seen_ids[1])}")

    return {
        "mot_rows":     mot_rows,
        "fps_avg":      fps_acc,
        "total_frames": frame_idx_,
        "unique_ids":   seen_ids,
    }
