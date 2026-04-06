"""
토마토 실시간 트래킹 - 단순화 버전
YOLO + ByteTrack + Stable ID

🔧 설정은 맨 위 CONFIG 딕셔너리에서만 수정!
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import supervision as sv
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO
from trackers import ByteTrackTracker


# ============================================================
# 🔧 설정 - 이것만 수정하면 됨!
# ============================================================
CONFIG = {
    # Detection
    "conf": 0.5,              # 검출 신뢰도 (낮을수록 많이 검출)
    "nms": 0.3,               # NMS threshold
    
    # ByteTrack  
    "byte_buffer": 300,       # 트랙 유지 프레임 수
    
    # ID 매칭 - 거리 기반 (픽셀)
    "center_max_dist": 500,   # 연속 프레임 중심점 최대 거리
    "reid_max_dist": 800,     # Re-ID 최대 거리
    "reid_max_frames": 150,   # Re-ID 시도 최대 프레임
    
    # ID 매칭 - 점수 기반
    "hist_weight": 0.4,       # HSV 히스토그램 비중 (0~1)
    "match_thresh": 0.25,     # 매칭 최소 점수 (낮을수록 관대)
    
    # 디버그
    "debug": False,
}
# ============================================================


CLASS_NAMES = {0: "ripe", 1: "unripe"}


@dataclass  
class TrackState:
    stable_id: int
    xyxy: np.ndarray
    class_id: int
    hist: Optional[np.ndarray]


class StableIdAssigner:
    """Stable ID 관리 - 단순화"""
    
    def __init__(self, debug: bool = False):
        self.debug = debug or CONFIG["debug"]
        self._next_id: Dict[int, int] = {}
        self._prev: List[TrackState] = []
        self._lost: List[dict] = []
    
    def reset(self):
        self._next_id.clear()
        self._prev = []
        self._lost = []
    
    def _new_id(self, class_id: int) -> int:
        n = self._next_id.get(class_id, 1)
        self._next_id[class_id] = n + 1
        return n
    
    @staticmethod
    def _compute_hist(frame: np.ndarray, xyxy: np.ndarray) -> Optional[np.ndarray]:
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
    def _center_dist(a: np.ndarray, b: np.ndarray) -> float:
        ax, ay = 0.5 * (a[0] + a[2]), 0.5 * (a[1] + a[3])
        bx, by = 0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3])
        return float(np.hypot(ax - bx, ay - by))
    
    def assign(self, frame_idx: int, frame: np.ndarray, dets: sv.Detections) -> np.ndarray:
        n = len(dets)
        
        # 빈 검출
        if n == 0:
            for p in self._prev:
                self._lost.append({
                    "stable_id": p.stable_id, "class_id": p.class_id,
                    "hist": p.hist, "xyxy": p.xyxy.copy(), "lost_at": frame_idx,
                })
            self._prev = []
            self._lost = [x for x in self._lost if frame_idx - x["lost_at"] < CONFIG["reid_max_frames"]]
            return np.array([], dtype=np.int32)
        
        xyxy = dets.xyxy
        classes = dets.class_id.astype(np.int32)
        hists = [self._compute_hist(frame, xyxy[i]) for i in range(n)]
        stable = np.full(n, -1, dtype=np.int32)
        matched_prev: Set[int] = set()
        
        # 1단계: 이전 프레임과 거리 기반 매칭 (클래스별)
        for cid in set(classes.tolist()):
            prev_idx = [i for i, p in enumerate(self._prev) if p.class_id == cid]
            det_idx = [j for j in range(n) if classes[j] == cid]
            
            if not prev_idx or not det_idx:
                continue
            
            # 거리 행렬
            cost = np.full((len(prev_idx), len(det_idx)), 9999.0)
            for ii, pi in enumerate(prev_idx):
                for jj, dj in enumerate(det_idx):
                    cost[ii, jj] = self._center_dist(self._prev[pi].xyxy, xyxy[dj])
            
            # 헝가리안 매칭
            ri, ci = linear_sum_assignment(cost)
            for ii, jj in zip(ri, ci):
                if cost[ii, jj] <= CONFIG["center_max_dist"]:
                    pi, dj = prev_idx[ii], det_idx[jj]
                    stable[dj] = self._prev[pi].stable_id
                    matched_prev.add(pi)
        
        # 2단계: 매칭 안 된 prev → lost
        for pi, p in enumerate(self._prev):
            if pi not in matched_prev:
                self._lost.append({
                    "stable_id": p.stable_id, "class_id": p.class_id,
                    "hist": p.hist, "xyxy": p.xyxy.copy(), "lost_at": frame_idx,
                })
        
        # 3단계: 매칭 안 된 det → lost에서 Re-ID
        for j in range(n):
            if stable[j] != -1:
                continue
            
            cid = int(classes[j])
            best_k, best_score = -1, -1.0
            
            for k, L in enumerate(self._lost):
                if L["class_id"] != cid:
                    continue
                if frame_idx - L["lost_at"] > CONFIG["reid_max_frames"]:
                    continue
                
                dist = self._center_dist(L["xyxy"], xyxy[j])
                if dist > CONFIG["reid_max_dist"]:
                    continue
                
                # 점수 계산
                pos_score = max(0, 1.0 - dist / CONFIG["reid_max_dist"])
                hist_score = 0.0
                if hists[j] is not None and L["hist"] is not None:
                    hist_score = max(0, float(cv2.compareHist(hists[j], L["hist"], cv2.HISTCMP_CORREL)))
                
                hw = CONFIG["hist_weight"]
                score = hw * hist_score + (1 - hw) * pos_score
                
                if score > best_score:
                    best_score = score
                    best_k = k
            
            if best_k >= 0 and best_score >= CONFIG["match_thresh"]:
                L = self._lost.pop(best_k)
                stable[j] = L["stable_id"]
        
        # 4단계: 여전히 -1 → 새 ID
        for j in range(n):
            if stable[j] == -1:
                cid = int(classes[j])
                new_id = self._new_id(cid)
                stable[j] = new_id
                
                if self.debug:
                    cx = 0.5 * (xyxy[j][0] + xyxy[j][2])
                    cy = 0.5 * (xyxy[j][1] + xyxy[j][3])
                    nearby = [(self._center_dist(L["xyxy"], xyxy[j]), L["stable_id"],
                              frame_idx - L["lost_at"])
                              for L in self._lost if L["class_id"] == cid]
                    nearby.sort()
                    info = ", ".join([f"#{s}({d:.0f}px,{a}f)" for d, s, a in nearby[:3]]) or "none"
                    print(f"[NEW] {CLASS_NAMES.get(cid)} #{new_id} at ({cx:.0f},{cy:.0f}) | lost: {info}")
        
        # prev 업데이트
        self._prev = [
            TrackState(stable_id=int(stable[j]), xyxy=xyxy[j].copy(),
                      class_id=int(classes[j]), hist=hists[j])
            for j in range(n)
        ]
        
        # 오래된 lost 제거
        self._lost = [x for x in self._lost if frame_idx - x["lost_at"] < CONFIG["reid_max_frames"]]
        
        return stable


def run(source: str = "0",
        model_path: str = r"runs\yolo26_custom_tomato\trained_yolo26_custom.pt",
        roi_half_width: Optional[int] = None,
        output_path: Optional[str] = None,
        show_window: bool = True,
        debug: bool = False):
    """
    토마토 트래킹 실행
    
    Args:
        source: 영상 소스 (0=웹캠, 또는 파일 경로)
        model_path: YOLO 모델 경로
        roi_half_width: ROI 반폭 (None=전체)
        output_path: 출력 파일 경로
        show_window: 창 표시
        debug: 디버그 출력
    """
    # 모델
    model = YOLO(model_path)
    
    # 영상
    vid = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(vid)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {source}")
    
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    # ROI
    roi = None
    if roi_half_width:
        cx = w // 2
        roi = (max(0, cx - roi_half_width), 0, min(w-1, cx + roi_half_width), h-1)
        print(f"[INFO] ROI: x=[{roi[0]}, {roi[2]}]")
    
    # Tracker
    tracker = ByteTrackTracker(
        minimum_consecutive_frames=1,
        lost_track_buffer=CONFIG["byte_buffer"],
        frame_rate=fps,
    )
    id_assigner = StableIdAssigner(debug=debug)
    
    # Annotators
    colors = sv.ColorPalette.from_hex(["#FF0000", "#00CC00"])
    box_ann = sv.BoxAnnotator(color=colors, color_lookup=sv.ColorLookup.CLASS)
    label_ann = sv.LabelAnnotator(color=colors, color_lookup=sv.ColorLookup.CLASS,
                                  text_color=sv.Color.WHITE, text_scale=0.6)
    trace_ann = sv.TraceAnnotator(color=colors, color_lookup=sv.ColorLookup.CLASS,
                                  thickness=2, trace_length=80)
    
    # Output
    writer = None
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    
    # 카운팅
    count_ripe, count_unripe = 0, 0
    counted: Set[Tuple[int, int]] = set()
    arm_left: Set[Tuple[int, int]] = set()
    arm_right: Set[Tuple[int, int]] = set()
    
    print(f"[INFO] Source: {source} ({w}x{h} @ {fps:.1f}fps)")
    print(f"[INFO] Model: {model_path}")
    print(f"[INFO] Config: conf={CONFIG['conf']}, center_dist={CONFIG['center_max_dist']}, reid_dist={CONFIG['reid_max_dist']}")
    
    fps_avg = 0.0
    frame_idx = 0
    
    while True:
        t0 = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detection
        if roi:
            roi_frame = frame[roi[1]:roi[3]+1, roi[0]:roi[2]+1]
            results = model(roi_frame, conf=CONFIG["conf"], verbose=False)[0]
            dets = sv.Detections.from_ultralytics(results).with_nms(threshold=CONFIG["nms"])
            if len(dets):
                dets.xyxy[:, [0, 2]] += roi[0]
                dets.xyxy[:, [1, 3]] += roi[1]
        else:
            results = model(frame, conf=CONFIG["conf"], verbose=False)[0]
            dets = sv.Detections.from_ultralytics(results).with_nms(threshold=CONFIG["nms"])
        
        # ByteTrack
        dets = tracker.update(dets)
        
        # 클래스 필터
        if dets.class_id is not None:
            valid = [i for i, c in enumerate(dets.class_id) if c in CLASS_NAMES]
            dets = dets[valid]
        
        # Stable ID
        stable_ids = id_assigner.assign(frame_idx, frame, dets)
        dets.tracker_id = stable_ids
        
        # ROI 카운팅
        if roi and len(dets):
            for i in range(len(dets)):
                sid = int(stable_ids[i])
                cid = int(dets.class_id[i])
                key = (cid, sid)
                
                if key in counted:
                    continue
                
                cx = 0.5 * (dets.xyxy[i][0] + dets.xyxy[i][2])
                
                if cx < roi[0]:
                    arm_left.add(key)
                elif cx > roi[2]:
                    arm_right.add(key)
                elif roi[0] <= cx <= roi[2]:
                    if key in arm_left or key in arm_right:
                        if cid == 0:
                            count_ripe += 1
                        else:
                            count_unripe += 1
                        counted.add(key)
                        arm_left.discard(key)
                        arm_right.discard(key)
        
        # 시각화
        vis = frame.copy()
        
        if roi:
            cv2.rectangle(vis, (roi[0], roi[1]), (roi[2], roi[3]), (255, 200, 0), 2)
        
        vis = box_ann.annotate(vis, dets)
        vis = trace_ann.annotate(vis, dets)
        
        labels = [f"{CLASS_NAMES.get(int(dets.class_id[i]))} #{int(dets.tracker_id[i])}"
                  for i in range(len(dets))]
        vis = label_ann.annotate(vis, dets, labels)
        
        # FPS
        dt = time.perf_counter() - t0
        fps_now = 1.0 / dt if dt > 0 else 0
        fps_avg = 0.1 * fps_now + 0.9 * fps_avg if fps_avg else fps_now
        
        # HUD
        n_r = int((dets.class_id == 0).sum()) if len(dets) else 0
        n_u = int((dets.class_id == 1).sum()) if len(dets) else 0
        hud = f"Count: {count_ripe}R/{count_unripe}U | Now: {n_r}R/{n_u}U | FPS: {fps_avg:.1f}"
        cv2.putText(vis, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if writer:
            writer.write(vis)
        
        if show_window:
            cv2.imshow("Tomato Tracker", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                tracker.reset()
                id_assigner.reset()
                count_ripe, count_unripe = 0, 0
                counted.clear()
                arm_left.clear()
                arm_right.clear()
                frame_idx = 0
                print("[RESET]")
        
        frame_idx += 1
    
    cap.release()
    if writer:
        writer.release()
    if show_window:
        cv2.destroyAllWindows()
    
    print(f"[DONE] ripe={count_ripe}, unripe={count_unripe}")
