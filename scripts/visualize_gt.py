#!/usr/bin/env python3
"""
gt_mot.txt → GT 확인 영상 생성

이미 만들어진 gt_mot.txt 를 프레임에 렌더링해서
ID가 올바른지 눈으로 확인하는 스크립트.

확인 포인트:
  - 같은 토마토에 항상 같은 색·번호가 붙는가?
  - 색/번호가 갑자기 바뀌는 프레임 = ID switching 오류

사용법:
    python scripts/visualize_gt.py
    python scripts/visualize_gt.py --gt tracking_result/gt_mot.txt --fps 5
    python scripts/visualize_gt.py --gt tracking_result/gt.csv --video notebook/rgb.mp4 --out tracking_result/gt_on_rgb.mp4
"""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2

REPO_ROOT = Path(__file__).resolve().parent.parent

CONFIG = {
    "mot_gt":    "tracking_result/gt_mot.csv",
    "frames_dir": "notebook/rgb_frames",
    "out_video": "tracking_result/gt_check.mp4",
    "fps":       10,   # 낮출수록 느리게 재생 → 확인하기 쉬움
}

CLASS_NAMES: Dict[int, str] = {0: "ripe", 1: "unripe"}
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _encode_h264(src: Path, dst: Path) -> bool:
    """ffmpeg으로 src(mp4v) → dst(H.264) 변환. 성공하면 True."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", str(src),
             "-vcodec", "libx264", "-crf", "18", "-preset", "fast",
             "-movflags", "+faststart", str(dst)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def _write_video(frames_iter, w: int, h: int, fps: int, out_path: Path):
    """프레임을 H.264 mp4로 저장. ffmpeg 없으면 mp4v fallback."""
    tmp = out_path.with_suffix(".tmp.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(tmp), fourcc, fps, (w, h))

    for frame in frames_iter:
        vw.write(frame)
    vw.release()

    if _encode_h264(tmp, out_path):
        tmp.unlink(missing_ok=True)
        print(f"  코덱: H.264 (libx264)")
    else:
        tmp.rename(out_path)
        print(f"  코덱: mp4v (ffmpeg 없음 — H.264 변환 생략)")


# BGR: ripe=빨간색, unripe=초록색
_CLASS_COLOR: Dict[int, Tuple[int, int, int]] = {
    0: (0,   50, 220),   # ripe   → 빨간색
    1: (0,  200,  50),   # unripe → 초록색
}


def load_mot(mot_path: Path) -> Dict[int, List]:
    """frame_id → [(track_id, x, y, w, h, class_id)]"""
    data: Dict[int, List] = {}
    with open(mot_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fid = int(row["frame_id"])
            data.setdefault(fid, []).append((
                int(row["track_id"]),
                float(row["x"]), float(row["y"]),
                float(row["w"]), float(row["h"]),
                int(row["class_id"]),
            ))
    return data


def _annotate_frame(frame, fid: int, frame_ann: Dict[int, List]) -> int:
    """박스·라벨을 frame 에 그립니다. 반환: 해당 프레임 박스 수."""
    anns = frame_ann.get(fid, [])
    for tid, x, y, w, h, cid in anns:
        color = _CLASS_COLOR.get(cid, (180, 180, 180))
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"{CLASS_NAMES.get(cid, str(cid))} #{tid}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    info = f"frame {fid}  |  boxes: {len(anns)}"
    cv2.putText(frame, info, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (220, 220, 220), 2, cv2.LINE_AA)
    return len(anns)


def render(mot_path: Path, frames_dir: Path, out_video: Path, fps: int) -> None:
    frame_ann = load_mot(mot_path)

    frame_paths = sorted(
        p for p in frames_dir.iterdir() if p.suffix.lower() in _IMG_EXTS
    )
    if not frame_paths:
        raise RuntimeError(f"프레임 없음: {frames_dir}")

    sample = cv2.imread(str(frame_paths[0]))
    if sample is None:
        raise RuntimeError(f"첫 프레임 읽기 실패: {frame_paths[0]}")
    h_img, w_img = sample.shape[:2]

    out_video.parent.mkdir(parents=True, exist_ok=True)

    all_tids = {tid for anns in frame_ann.values() for tid, *_ in anns}
    print(f"[visualize_gt] 총 track ID 수: {len(all_tids)}  프레임: {len(frame_paths)}")

    def _frame_gen():
        for frame_idx, frame_path in enumerate(frame_paths):
            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue

            fid = frame_idx + 1
            _annotate_frame(frame, fid, frame_ann)

            if (frame_idx + 1) % 100 == 0 or frame_idx + 1 == len(frame_paths):
                print(f"  [{frame_idx+1:4d}/{len(frame_paths)}]")

            yield frame

    _write_video(_frame_gen(), w_img, h_img, fps, out_video)
    print(f"\n저장 완료: {out_video}")
    print("확인 포인트:")
    print("  · 같은 토마토 → 항상 같은 색/번호  (OK)")
    print("  · 색·번호가 갑자기 바뀌는 순간      → ID switching 오류")
    print("  · 박스가 없는데 토마토가 있는 순간   → FN (missed detection)")


def render_from_video(
    mot_path: Path,
    video_path: Path,
    out_video: Path,
    fps: Optional[float],
) -> None:
    """입력 영상에 MOT 박스를 그려 출력 (frame_id 는 1-based, 영상 프레임 순서와 일치)."""
    frame_ann = load_mot(mot_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {video_path}")

    vfps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out_fps = float(fps) if fps is not None else vfps
    out_fps = max(1.0, out_fps)

    w_img = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_img = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_cap = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    out_video.parent.mkdir(parents=True, exist_ok=True)
    all_tids = {tid for anns in frame_ann.values() for tid, *_ in anns}
    max_gt_fid = max(frame_ann.keys()) if frame_ann else 0
    print(f"[visualize_gt] video: {video_path.name}  {w_img}x{h_img}  source_fps={vfps:.2f}  out_fps={out_fps:.2f}")
    print(f"[visualize_gt] 총 track ID 수: {len(all_tids)}  GT max frame_id: {max_gt_fid}  cap_frames~{n_cap}")

    def _frame_gen():
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            fid = frame_idx
            _annotate_frame(frame, fid, frame_ann)
            if frame_idx % 100 == 0:
                print(f"  [{frame_idx:4d}]")
            yield frame

    try:
        _write_video(_frame_gen(), w_img, h_img, int(round(out_fps)), out_video)
    finally:
        cap.release()
    print(f"\n저장 완료: {out_video}")
    print("확인 포인트:")
    print("  · 같은 토마토 → 항상 같은 색/번호  (OK)")
    print("  · 색·번호가 갑자기 바뀌는 순간      → ID switching 오류")
    print("  · 박스가 없는데 토마토가 있는 순간   → FN (missed detection)")


def main() -> None:
    p = argparse.ArgumentParser(description="gt_mot.txt → GT 확인 영상")
    p.add_argument("--gt",     type=str, default=None, help="MOT GT 파일 경로")
    p.add_argument("--frames", type=str, default=None, help="프레임 디렉터리 (--video 없을 때)")
    p.add_argument("--video",  type=str, default=None, help="입력 영상 (지정 시 프레임 폴더 대신 사용)")
    p.add_argument("--out",    type=str, default=None, help="출력 영상 경로")
    p.add_argument("--fps",    type=float, default=None,
                   help="출력 FPS (--video: 기본=입력 영상 FPS, 프레임 모드: 기본=10)")
    args = p.parse_args()

    cfg = dict(CONFIG)
    if args.gt:     cfg["mot_gt"]     = args.gt
    if args.frames: cfg["frames_dir"] = args.frames
    if args.out:    cfg["out_video"]  = args.out
    if args.fps is not None and not args.video:
        cfg["fps"] = int(args.fps)

    if args.video:
        out = REPO_ROOT / (args.out or "tracking_result/gt_on_rgb.mp4")
        render_from_video(
            mot_path   = REPO_ROOT / cfg["mot_gt"],
            video_path = REPO_ROOT / args.video,
            out_video  = out,
            fps        = args.fps,
        )
        return

    fps = args.fps if args.fps is not None else float(cfg["fps"])
    render(
        mot_path   = REPO_ROOT / cfg["mot_gt"],
        frames_dir = REPO_ROOT / cfg["frames_dir"],
        out_video  = REPO_ROOT / cfg["out_video"],
        fps        = int(round(fps)),
    )


if __name__ == "__main__":
    main()
