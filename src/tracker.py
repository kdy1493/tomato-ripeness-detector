"""
토마토 실시간 트래킹 - 단순화 버전
YOLO + ByteTrack + Stable ID

설정은 맨 위 CONFIG 딕셔너리에서만 수정!
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
from trackers import ByteTrackTracker


# ============================================================
# 설정 - 이것만 수정하면 됨!
# ============================================================
CONFIG = {
    # Detection
    "conf": 0.5,              # 검출 신뢰도 (낮을수록 많이 검출)
    "nms": 0.3,               # NMS threshold

    # ByteTrack
    "byte_buffer": 300,       # 트랙 유지 프레임 수

    # ROI 내 연속 프레임 ID 유지 (픽셀)
    "center_max_dist": 500,   # 연속 프레임 중심점 최대 거리

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


class StableIdAssigner:
    """Stable ID 관리.

    ROI 내부 객체에만 ID를 발급한다 (외부는 -1).
    한번 발급된 ID는 재사용하지 않는다.
    ROI를 벗어난 객체는 ID가 소멸되며 재진입 시 새 ID를 받는다.
    """

    def __init__(self, debug: bool = False):
        self.debug = debug or CONFIG["debug"]
        self._next_id: Dict[int, int] = {}
        self._prev: List[TrackState] = []

    def reset(self):
        self._next_id.clear()
        self._prev = []

    def _new_id(self, class_id: int) -> int:
        n = self._next_id.get(class_id, 1)
        self._next_id[class_id] = n + 1
        return n

    @staticmethod
    def _center_dist(a: np.ndarray, b: np.ndarray) -> float:
        ax, ay = 0.5 * (a[0] + a[2]), 0.5 * (a[1] + a[3])
        bx, by = 0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3])
        return float(np.hypot(ax - bx, ay - by))

    def assign(self, frame_idx: int, dets: sv.Detections,
               roi: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """stable_id 배열 반환. ROI 외부 객체는 -1."""
        n = len(dets)

        if n == 0:
            self._prev = []
            return np.array([], dtype=np.int32)

        xyxy = dets.xyxy
        classes = dets.class_id.astype(np.int32)
        stable = np.full(n, -1, dtype=np.int32)

        # ROI 마스크: roi가 없으면 전체 프레임이 대상
        if roi is not None:
            centers_x = 0.5 * (xyxy[:, 0] + xyxy[:, 2])
            in_roi = (centers_x >= roi[0]) & (centers_x <= roi[2])
        else:
            in_roi = np.ones(n, dtype=bool)

        roi_idx = [j for j in range(n) if in_roi[j]]
        matched_prev: Set[int] = set()

        # 1단계: ROI 내 객체 ↔ 이전 프레임 매칭으로 ID 유지 (클래스별 헝가리안)
        for cid in set(int(classes[j]) for j in roi_idx):
            prev_idx = [i for i, p in enumerate(self._prev) if p.class_id == cid]
            det_idx = [j for j in roi_idx if classes[j] == cid]

            if not prev_idx or not det_idx:
                continue

            cost = np.full((len(prev_idx), len(det_idx)), 9999.0)
            for ii, pi in enumerate(prev_idx):
                for jj, dj in enumerate(det_idx):
                    cost[ii, jj] = self._center_dist(self._prev[pi].xyxy, xyxy[dj])

            ri, ci = linear_sum_assignment(cost)
            for ii, jj in zip(ri, ci):
                if cost[ii, jj] <= CONFIG["center_max_dist"]:
                    pi, dj = prev_idx[ii], det_idx[jj]
                    stable[dj] = self._prev[pi].stable_id
                    matched_prev.add(pi)

        # 2단계: ROI 진입 신규 객체 → 새 ID 발급 (우상단→좌하단 순서)
        new_entries = [
            (j, int(classes[j])) for j in roi_idx if stable[j] == -1
        ]
        new_entries.sort(key=lambda item: (
            item[1],
            (0.5 * (xyxy[item[0]][1] + xyxy[item[0]][3]))
            - (0.5 * (xyxy[item[0]][0] + xyxy[item[0]][2]))
        ))

        for j, cid in new_entries:
            new_id = self._new_id(cid)
            stable[j] = new_id

            if self.debug:
                cx = 0.5 * (xyxy[j][0] + xyxy[j][2])
                cy = 0.5 * (xyxy[j][1] + xyxy[j][3])
                print(f"[NEW] {CLASS_NAMES.get(cid)} #{new_id} at ({cx:.0f},{cy:.0f})")

        # prev 업데이트: ROI 내 객체만 유지 (ROI 이탈 시 자동 소멸)
        self._prev = [
            TrackState(stable_id=int(stable[j]), xyxy=xyxy[j].copy(), class_id=int(classes[j]))
            for j in roi_idx if stable[j] != -1
        ]

        return stable


_DEFAULT_MODEL = str(
    Path(__file__).parent.parent / "runs" / "yolo26_custom_tomato" / "trained_yolo26_custom.pt"
)


def run(source: str = "0",
        model_path: str = _DEFAULT_MODEL,
        roi_half_width: Optional[int] = None,
        output_path: Optional[str] = None,
        show_window: bool = True,
        debug: bool = False,
        save_results: Optional[str] = None):
    """
    토마토 트래킹 실행

    Args:
        source: 영상 소스 (0=웹캠, 또는 파일 경로)
        model_path: YOLO 모델 경로
        roi_half_width: ROI 반폭 (None=전체 프레임 사용)
        output_path: 출력 영상 파일 경로
        show_window: 창 표시
        debug: 디버그 출력
        save_results: 결과 저장 기본 경로 (예: "out/result" → result.csv, result.json 생성)
    """
    model = YOLO(model_path)

    vid = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(vid)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {source}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # ROI: 화면 중앙 수직 스트립
    roi = None
    if roi_half_width:
        cx = w // 2
        roi = (max(0, cx - roi_half_width), 0, min(w - 1, cx + roi_half_width), h - 1)
        print(f"[INFO] ROI: x=[{roi[0]}, {roi[2]}]")

    tracker = ByteTrackTracker(
        minimum_consecutive_frames=1,
        lost_track_buffer=CONFIG["byte_buffer"],
        frame_rate=fps,
    )
    id_assigner = StableIdAssigner(debug=debug)

    colors = sv.ColorPalette.from_hex(["#FF0000", "#00CC00"])
    box_ann = sv.BoxAnnotator(color=colors, color_lookup=sv.ColorLookup.CLASS)
    label_ann = sv.LabelAnnotator(color=colors, color_lookup=sv.ColorLookup.CLASS,
                                  text_color=sv.Color.WHITE, text_scale=0.6)
    trace_ann = sv.TraceAnnotator(color=colors, color_lookup=sv.ColorLookup.CLASS,
                                  thickness=2, trace_length=80)

    writer = None
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    # 카운팅: 새 stable_id 발급 = ROI 진입 = 카운트
    count_ripe, count_unripe = 0, 0
    counted: Set[Tuple[int, int]] = set()   # (class_id, stable_id)
    frame_log: List[Tuple] = []

    print(f"[INFO] Source: {source} ({w}x{h} @ {fps:.1f}fps)")
    print(f"[INFO] Model: {model_path}")
    print(f"[INFO] Config: conf={CONFIG['conf']}, center_dist={CONFIG['center_max_dist']}")

    fps_avg = 0.0
    frame_idx = 0

    while True:
        t0 = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break

        # 검출: 항상 전체 프레임 (ROI는 ID 발급/카운팅 경계로만 사용)
        results = model(frame, conf=CONFIG["conf"], verbose=False)[0]
        dets = sv.Detections.from_ultralytics(results).with_nms(threshold=CONFIG["nms"])

        dets = tracker.update(dets)

        if dets.class_id is not None:
            valid = [i for i, c in enumerate(dets.class_id) if c in CLASS_NAMES]
            dets = dets[valid]

        # ByteTrack ID 저장 (궤적용) — stable_id로 덮어쓰기 전에 보존
        bytetrack_ids = dets.tracker_id.copy() if dets.tracker_id is not None else None

        # ROI 내 객체에만 stable_id 발급 (외부는 -1)
        stable_ids = id_assigner.assign(frame_idx, dets, roi=roi)

        # 카운팅: stable_id가 새로 발급된 객체 (= ROI 최초 진입)
        for i in range(len(dets)):
            sid = int(stable_ids[i])
            if sid == -1:
                continue
            cid = int(dets.class_id[i])
            key = (cid, sid)
            if key not in counted:
                if cid == 0:
                    count_ripe += 1
                else:
                    count_unripe += 1
                counted.add(key)

        # 시각화
        vis = frame.copy()

        if roi:
            cv2.rectangle(vis, (roi[0], roi[1]), (roi[2], roi[3]), (255, 200, 0), 2)

        vis = box_ann.annotate(vis, dets)

        # 궤적: ByteTrack ID 기반 (객체별 연속 히스토리 유지, -1 혼용 없음)
        dets.tracker_id = bytetrack_ids
        vis = trace_ann.annotate(vis, dets)

        # 레이블은 stable_id 사용
        dets.tracker_id = stable_ids

        # ROI 외부(sid=-1)는 클래스명만, 내부는 클래스명 + ID
        labels = []
        for i in range(len(dets)):
            cname = CLASS_NAMES.get(int(dets.class_id[i]), "?")
            sid = int(dets.tracker_id[i])
            labels.append(cname if sid == -1 else f"{cname} #{sid}")
        vis = label_ann.annotate(vis, dets, labels)

        dt = time.perf_counter() - t0
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
                tracker.reset()
                id_assigner.reset()
                count_ripe, count_unripe = 0, 0
                counted.clear()
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
        csv_path = base.with_suffix(".csv")
        json_path = base.with_suffix(".json")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            cw = csv.writer(f)
            cw.writerow(["frame", "n_ripe", "n_unripe", "total_ripe", "total_unripe"])
            cw.writerows(frame_log)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {"total_ripe": count_ripe, "total_unripe": count_unripe, "total_frames": frame_idx},
                f, indent=2,
            )
        print(f"[SAVED] {csv_path}, {json_path}")
