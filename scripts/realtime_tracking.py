"""
YOLO26 + ByteTrack 실시간 트래킹 실행 스크립트.

사용법:
    python scripts/realtime_tracking.py
    python scripts/realtime_tracking.py --source rtsp://192.168.0.100:554/stream
    python scripts/realtime_tracking.py --compensation
    python scripts/realtime_tracking.py --model path/to/best.pt --conf 0.3
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.realtime_tracking import run


def main():
    parser = argparse.ArgumentParser(description="YOLO26 실시간 트래킹")
    parser.add_argument("--source", default="0",
                        help="웹캠 인덱스 (0,1,...) 또는 RTSP/파일 경로")
    parser.add_argument("--model", default=None,
                        help="YOLO 모델 가중치 경로")
    parser.add_argument("--conf", type=float, default=0.2,
                        help="검출 신뢰도 임계값")
    parser.add_argument("--nms", type=float, default=0.3,
                        help="NMS IoU 임계값")
    parser.add_argument("--compensation", action="store_true",
                        help="카메라 모션 보정 활성화 (이동 카메라용)")
    parser.add_argument("--min-frames", type=int, default=1,
                        help="트랙 확정까지 최소 연속 프레임 수")
    args = parser.parse_args()

    kwargs = dict(
        source=args.source,
        conf=args.conf,
        nms=args.nms,
        compensation=args.compensation,
        min_frames=args.min_frames,
    )
    if args.model is not None:
        kwargs["model_path"] = args.model

    run(**kwargs)


if __name__ == "__main__":
    main()
