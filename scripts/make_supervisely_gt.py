#!/usr/bin/env python3
"""
YOLO + ByteTrack → Supervisely 비디오 어노테이션 GT 자동 생성

흐름:
  1. notebook/rgb_frames/ 이미지를 이름순 정렬
  2. YOLO detection → 클래스별 독립 ByteTrack (stable ID 유지)
  3. Supervisely 비디오 어노테이션 JSON 생성
  4. MOT Challenge CSV 동시 생성 (benchmark용)

출력:
  tracking_result/supervisely_gt/
    meta.json          ← 프로젝트 클래스 정의
    ann.json           ← 비디오 어노테이션 (track ID 포함)
  tracking_result/gt_mot.csv   ← benchmark.py --gt 용 MOT GT

Supervisely 업로드:
  1. app.supervisely.com → 새 프로젝트 (Videos)
  2. 비디오 업로드 (notebook/rgb.mp4 또는 rgb_frames를 mp4로 변환)
  3. 프로젝트 설정 → Import → "Supervisely" 포맷 → ann.json 업로드

검수 후 MOT 변환:
  python scripts/sly2mot.py   (Supervisely에서 export한 JSON → gt_mot.csv)

사용법:
    python scripts/make_supervisely_gt.py
    python scripts/make_supervisely_gt.py --conf 0.25 --viz
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import supervision as sv
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# ============================================================
# 설정  ← 이것만 수정
# ============================================================
CONFIG: dict = {
    "frames_dir":  "notebook/rgb_frames",
    "model_path":  "runs/yolo26_custom_tomato/trained_yolo26_custom.pt",
    "output_dir":  "tracking_result/supervisely_gt",
    "mot_gt_path": "tracking_result/gt_mot.csv",
    "video_path":  "notebook/rgb.mp4",   # 원본 비디오 (ds0/video/ 에 복사됨)
    "video_name":  "rgb.mp4",            # ann 파일명에 사용 (video_path의 파일명과 일치해야 함)

    "conf": 0.25,
    "iou":  0.3,

    "track_activation_threshold": 0.25,
    "lost_track_buffer":          30,
    "minimum_matching_threshold": 0.8,
    "frame_rate":                 30,

    "viz":        False,
    "viz_output": "tracking_result/gt_check.mp4",
    "viz_fps":    10,
}
# ============================================================

CLASS_NAMES:  Dict[int, str] = {0: "ripe",    1: "unripe"}
CLASS_COLORS: Dict[int, str] = {0: "#FF5050", 1: "#50C878"}

# ripe=빨간, unripe=초록 (BGR)
_CLASS_COLOR_BGR: Dict[int, Tuple[int, int, int]] = {
    0: (0,  50, 220),
    1: (0, 200,  50),
}

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
_NOW = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


# ---------------------------------------------------------------------------
# 유틸
# ---------------------------------------------------------------------------

def collect_frames(frames_dir: Path) -> List[Path]:
    return sorted(p for p in frames_dir.iterdir() if p.suffix.lower() in _IMG_EXTS)


# ---------------------------------------------------------------------------
# Supervisely JSON 빌더
# ---------------------------------------------------------------------------

def build_meta() -> dict:
    """meta.json: 프로젝트 클래스 정의."""
    return {
        "classes": [
            {
                "title":  CLASS_NAMES[cid],
                "shape":  "rectangle",
                "color":  CLASS_COLORS[cid],
                "hotkey": "",
            }
            for cid in sorted(CLASS_NAMES)
        ],
        "tags":        [],
        "projectType": "videos",
    }


def build_ann(
    tracks: Dict[Tuple[int, int], List[Tuple[int, float, float, float, float]]],
    n_frames: int,
    img_w: int,
    img_h: int,
) -> dict:
    """ann.json: Supervisely 비디오 어노테이션 (공식 포맷).

    tracks: {(stable_id, class_id): [(frame_idx, x1, y1, x2, y2), ...]}

    Supervisely video 포맷 핵심:
      - objects[].key          (문자열, 전역 고유)
      - figures[].objectKey    (objects[].key 참조)
      - figures[].geometryType = "rectangle"
      - figures[].classTitle   (클래스 이름)
    """
    frames_dict: Dict[int, List[dict]] = {i: [] for i in range(n_frames)}
    objects = []

    for (sid, cid), boxes in sorted(tracks.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        obj_key = str(uuid.uuid4())
        objects.append({
            "key":          obj_key,
            "classTitle":   CLASS_NAMES[cid],
            "tags":         [],
            "labelerLogin": "auto",
            "createdAt":    _NOW,
            "updatedAt":    _NOW,
        })

        for frame_idx, x1, y1, x2, y2 in sorted(boxes, key=lambda b: b[0]):
            frames_dict[frame_idx].append({
                "key":          str(uuid.uuid4()),
                "objectKey":    obj_key,
                "classTitle":   CLASS_NAMES[cid],
                "geometryType": "rectangle",
                "geometry": {
                    "points": {
                        "exterior": [[round(x1), round(y1)],
                                     [round(x2), round(y2)]],
                        "interior": [],
                    }
                },
                "labelerLogin": "auto",
                "createdAt":    _NOW,
                "updatedAt":    _NOW,
                "tags": [],
            })

    frames_list = [
        {"index": i, "figures": figures}
        for i, figures in frames_dict.items()
        if figures
    ]

    return {
        "description": "auto-generated by make_supervisely_gt.py",
        "size":        {"width": img_w, "height": img_h},
        "framesCount": n_frames,
        "tags":        [],
        "objects":     objects,
        "frames":      frames_list,
    }


# ---------------------------------------------------------------------------
# MOT CSV 저장
# ---------------------------------------------------------------------------

def write_mot_csv(
    tracks: Dict[Tuple[int, int], List[Tuple[int, float, float, float, float]]],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for (sid, cid), boxes in tracks.items():
        for frame_idx, x1, y1, x2, y2 in boxes:
            rows.append((
                frame_idx + 1,          # 1-based frame_id
                sid,                    # track_id
                round(x1, 2),
                round(y1, 2),
                round(x2 - x1, 2),     # width
                round(y2 - y1, 2),     # height
                1,                     # conf
                cid,                   # class_id
            ))
    rows.sort(key=lambda r: (r[0], r[7], r[1]))
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame_id", "track_id", "x", "y", "w", "h", "conf", "class_id"])
        w.writerows(rows)


# ---------------------------------------------------------------------------
# GT 확인 영상
# ---------------------------------------------------------------------------

def _encode_h264(src: Path, dst: Path) -> bool:
    try:
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", str(src),
             "-vcodec", "libx264", "-crf", "18", "-preset", "fast",
             "-movflags", "+faststart", str(dst)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return r.returncode == 0
    except FileNotFoundError:
        return False


def render_gt_video(
    tracks: Dict[Tuple[int, int], List[Tuple[int, float, float, float, float]]],
    frame_paths: List[Path],
    out_video: Path,
    fps: int = 10,
) -> None:
    if not frame_paths:
        return
    sample = cv2.imread(str(frame_paths[0]))
    if sample is None:
        return
    h_img, w_img = sample.shape[:2]

    # frame_idx → [(cid, x1,y1,x2,y2, sid)]
    frame_map: Dict[int, List] = {}
    for (sid, cid), boxes in tracks.items():
        for frame_idx, x1, y1, x2, y2 in boxes:
            frame_map.setdefault(frame_idx, []).append((cid, x1, y1, x2, y2, sid))

    out_video.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_video.with_suffix(".tmp.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(tmp), fourcc, fps, (w_img, h_img))

    for frame_idx, frame_path in enumerate(frame_paths):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue
        for cid, x1, y1, x2, y2, sid in frame_map.get(frame_idx, []):
            color = _CLASS_COLOR_BGR.get(cid, (180, 180, 180))
            ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), color, 2)
            label = f"{CLASS_NAMES[cid]} #{sid}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (ix1, iy1 - th - 6), (ix1 + tw + 4, iy1), color, -1)
            cv2.putText(frame, label, (ix1 + 2, iy1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"frame {frame_idx+1}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)
        vw.write(frame)

    vw.release()
    if _encode_h264(tmp, out_video):
        tmp.unlink(missing_ok=True)
        print(f"  코덱: H.264 (libx264)")
    else:
        tmp.rename(out_video)
        print(f"  코덱: mp4v (ffmpeg 없음)")
    print(f"[viz] GT 확인 영상: {out_video}")


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def run(config: dict) -> None:
    frames_dir  = REPO_ROOT / config["frames_dir"]
    frame_paths = collect_frames(frames_dir)
    if not frame_paths:
        raise RuntimeError(f"이미지 없음: {frames_dir}")

    n_frames = len(frame_paths)
    print(f"[make_supervisely_gt] 프레임: {n_frames}장")

    # 해상도를 비디오 파일에서 직접 읽음 (frames 폴더 이미지보다 정확)
    src_video = REPO_ROOT / config["video_path"]
    if src_video.exists():
        cap = cv2.VideoCapture(str(src_video))
        img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        print(f"  비디오 해상도: {img_w}x{img_h}")
    else:
        sample = cv2.imread(str(frame_paths[0]))
        if sample is None:
            raise RuntimeError("첫 프레임 읽기 실패")
        img_h, img_w = sample.shape[:2]
        print(f"  프레임 해상도: {img_w}x{img_h} (비디오 파일 없음)")

    model = YOLO(str(REPO_ROOT / config["model_path"]))

    trackers: Dict[int, sv.ByteTrack] = {
        cid: sv.ByteTrack(
            track_activation_threshold=config["track_activation_threshold"],
            lost_track_buffer=config["lost_track_buffer"],
            minimum_matching_threshold=config["minimum_matching_threshold"],
            frame_rate=config["frame_rate"],
        )
        for cid in CLASS_NAMES
    }

    raw_to_stable: Dict[int, Dict[int, int]] = {cid: {} for cid in CLASS_NAMES}
    next_sid:      Dict[int, int]            = {cid: 1  for cid in CLASS_NAMES}

    # {(stable_id, class_id): [(frame_idx, x1, y1, x2, y2)]}
    tracks: Dict[Tuple[int, int], List] = {}

    for frame_idx, frame_path in enumerate(frame_paths):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            print(f"  [WARN] 읽기 실패: {frame_path.name}")
            continue

        results  = model(frame, conf=config["conf"], iou=config["iou"], verbose=False)[0]
        all_dets = sv.Detections.from_ultralytics(results)

        for cid, tracker in trackers.items():
            mask     = all_dets.class_id == cid
            cls_dets = all_dets[mask]
            if len(cls_dets) == 0:
                continue
            cls_dets = tracker.update_with_detections(cls_dets)
            if cls_dets.tracker_id is None or len(cls_dets) == 0:
                continue

            # 신규 raw ID → stable ID (cy-cx 오름차순)
            new_idx = [i for i, r in enumerate(cls_dets.tracker_id)
                       if int(r) not in raw_to_stable[cid]]
            new_idx.sort(key=lambda i: (
                0.5 * (cls_dets.xyxy[i][1] + cls_dets.xyxy[i][3])
                - 0.5 * (cls_dets.xyxy[i][0] + cls_dets.xyxy[i][2])
            ))
            for i in new_idx:
                raw_to_stable[cid][int(cls_dets.tracker_id[i])] = next_sid[cid]
                next_sid[cid] += 1

            for i in range(len(cls_dets)):
                sid             = raw_to_stable[cid][int(cls_dets.tracker_id[i])]
                x1, y1, x2, y2 = cls_dets.xyxy[i]
                tracks.setdefault((sid, cid), []).append(
                    (frame_idx, float(x1), float(y1), float(x2), float(y2))
                )

        if (frame_idx + 1) % 100 == 0 or frame_idx + 1 == n_frames:
            n_r = sum(1 for _, c in tracks if c == 0)
            n_u = sum(1 for _, c in tracks if c == 1)
            print(f"  [{frame_idx+1:4d}/{n_frames}]  tracks  ripe={n_r}  unripe={n_u}")

    # ── Supervisely 파일 저장 ────────────────────────────────
    # 필수 구조:
    #   supervisely_gt/
    #   ├── meta.json
    #   └── ds0/
    #       ├── ann/<video_name>.json
    #       └── video/<video_name>        ← 실제 비디오 파일 필수
    video_name = config.get("video_name", "rgb.mp4")
    out_dir    = REPO_ROOT / config["output_dir"]
    ann_dir    = out_dir / "ds0" / "ann"
    vid_dir    = out_dir / "ds0" / "video"
    ann_dir.mkdir(parents=True, exist_ok=True)
    vid_dir.mkdir(parents=True, exist_ok=True)

    meta_dict = build_meta()
    ann_dict  = build_ann(tracks, n_frames, img_w, img_h)

    meta_path = out_dir / "meta.json"
    ann_path  = ann_dir / f"{video_name}.json"
    meta_path.write_text(json.dumps(meta_dict, ensure_ascii=False, indent=2), encoding="utf-8")
    ann_path.write_text( json.dumps(ann_dict,  ensure_ascii=False, indent=2), encoding="utf-8")

    # 비디오 파일 복사
    src_video = REPO_ROOT / config["video_path"]
    dst_video = vid_dir / video_name
    if src_video.exists():
        if not dst_video.exists():
            shutil.copy2(src_video, dst_video)
            print(f"  비디오 복사: {dst_video}")
        else:
            print(f"  비디오 이미 존재: {dst_video}")
    else:
        print(f"  [WARN] 비디오 파일 없음: {src_video}")
        print(f"         직접 {dst_video} 에 비디오를 복사하세요.")

    # ── MOT CSV 저장 ─────────────────────────────────────────
    mot_path = REPO_ROOT / config["mot_gt_path"]
    write_mot_csv(tracks, mot_path)

    n_r = sum(1 for _, c in tracks if c == 0)
    n_u = sum(1 for _, c in tracks if c == 1)
    print(f"\n총 track: ripe={n_r}  unripe={n_u}  (total={n_r + n_u})")
    print(f"Supervisely 저장: {out_dir}")
    print(f"  meta.json             : {meta_path}")
    print(f"  ds0/ann/{video_name}.json  : {ann_path}")
    print(f"  ds0/video/{video_name}     : {dst_video}")
    print(f"MOT GT 저장: {mot_path}")

    # ── GT 확인 영상 ─────────────────────────────────────────
    if config.get("viz"):
        render_gt_video(
            tracks=tracks,
            frame_paths=frame_paths,
            out_video=REPO_ROOT / config["viz_output"],
            fps=config.get("viz_fps", 10),
        )

    print(f"""
─── Supervisely 업로드 방법 ────────────────────────────────
 1. app.supervisely.com → Import → "Supervisely" 포맷 선택
 2. supervisely_gt/ 폴더 통째로 업로드

    폴더 구조 확인:
      supervisely_gt/
      ├── meta.json
      └── ds0/
          ├── ann/{video_name}.json
          └── video/{video_name}   ← 비디오 파일 필수

 검수 완료 후 MOT 변환:
   python scripts/sly2mot.py
────────────────────────────────────────────────────────────""")


def main() -> None:
    p = argparse.ArgumentParser(description="YOLO + ByteTrack → Supervisely GT")
    p.add_argument("--frames",  type=str,   default=None)
    p.add_argument("--model",   type=str,   default=None)
    p.add_argument("--out",     type=str,   default=None)
    p.add_argument("--mot-gt",    type=str,   default=None, dest="mot_gt_path")
    p.add_argument("--video-path", type=str, default=None, dest="video_path",
                   help="원본 비디오 경로 (repo-relative, 기본: notebook/rgb.mp4)")
    p.add_argument("--video-name", type=str, default=None, dest="video_name",
                   help="비디오 파일명 (기본: rgb.mp4, video_path와 일치해야 함)")
    p.add_argument("--conf",    type=float, default=None)
    p.add_argument("--iou",     type=float, default=None)
    p.add_argument("--viz",     action="store_true")
    p.add_argument("--viz-fps", type=int,   default=None, dest="viz_fps")
    args = p.parse_args()

    cfg = dict(CONFIG)
    if args.frames:      cfg["frames_dir"]  = args.frames
    if args.model:       cfg["model_path"]  = args.model
    if args.out:         cfg["output_dir"]  = args.out
    if args.mot_gt_path: cfg["mot_gt_path"] = args.mot_gt_path
    if args.video_path:  cfg["video_path"]  = args.video_path
    if args.video_name:  cfg["video_name"]  = args.video_name
    if args.conf is not None: cfg["conf"]   = args.conf
    if args.iou  is not None: cfg["iou"]    = args.iou
    if args.viz:         cfg["viz"]         = True
    if args.viz_fps:     cfg["viz_fps"]     = args.viz_fps

    run(cfg)


if __name__ == "__main__":
    main()
