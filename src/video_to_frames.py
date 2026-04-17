#!/usr/bin/env python3
"""
영상을 프레임 이미지로 저장 (MOT GT 작업용)

기본: notebook/rgb.mp4 -> notebook/rgb_frames/rgb_000001.jpg ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parent.parent


def extract_frames(
    video_path: Path,
    out_dir: Path,
    *,
    prefix: str | None = None,
    ext: str = ".jpg",
    jpeg_quality: int = 95,
    stride: int = 1,
    start_frame: int = 0,
    max_frames: int | None = None,
) -> int:
    """
    Returns number of images written.
    """
    if stride < 1:
        raise ValueError("stride must be >= 1")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {video_path}")

    stem = video_path.stem
    name_prefix = prefix if prefix is not None else stem
    out_dir.mkdir(parents=True, exist_ok=True)

    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    print(f"video: {video_path}")
    print(f"size: {w}x{h}, fps: {fps:.3f}, frames: {total}, fourcc: {fourcc}")

    encode_params = []
    ext_l = ext.lower()
    if ext_l in (".jpg", ".jpeg"):
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]

    idx = 0
    written = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx < start_frame:
            idx += 1
            continue
        if (idx - start_frame) % stride != 0:
            idx += 1
            continue
        if max_frames is not None and written >= max_frames:
            break

        out_path = out_dir / f"{name_prefix}_{written + 1:06d}{ext_l}"
        if not cv2.imwrite(str(out_path), frame, encode_params):
            cap.release()
            raise RuntimeError(f"이미지 저장 실패: {out_path}")
        written += 1
        idx += 1

    cap.release()
    return written


def main() -> int:
    p = argparse.ArgumentParser(description="영상 -> 프레임 이미지")
    p.add_argument(
        "--video",
        type=Path,
        default=REPO_ROOT / "notebook" / "rgb.mp4",
        help="입력 영상 경로 (기본: repo/notebook/rgb.mp4)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="출력 폴더 (기본: notebook/<영상파일명>_frames)",
    )
    p.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="파일명 접두사 (기본: 영상 stem, 예: rgb -> rgb_000001.jpg)",
    )
    p.add_argument("--stride", type=int, default=1, help="N프레임마다 1장 (기본: 매 프레임)")
    p.add_argument("--start", type=int, default=0, help="시작 프레임 인덱스 (0부터)")
    p.add_argument("--max", type=int, default=None, help="저장할 최대 장 수")
    p.add_argument("--ext", type=str, default=".jpg", choices=[".jpg", ".jpeg", ".png"])
    p.add_argument("--jpeg-quality", type=int, default=95, help="JPEG 품질 0-100")
    args = p.parse_args()

    video_path = args.video
    if not video_path.is_absolute():
        video_path = (REPO_ROOT / video_path).resolve()

    out_dir = args.out
    if out_dir is None:
        out_dir = video_path.parent / f"{video_path.stem}_frames"
    elif not out_dir.is_absolute():
        out_dir = (REPO_ROOT / out_dir).resolve()

    n = extract_frames(
        video_path,
        out_dir,
        prefix=args.prefix,
        ext=args.ext,
        jpeg_quality=args.jpeg_quality,
        stride=args.stride,
        start_frame=args.start,
        max_frames=args.max,
    )
    print(f"saved {n} frames -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
