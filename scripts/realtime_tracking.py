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

from src.tracker import run


def main():
    parser = argparse.ArgumentParser(description="ByteTrack 실시간 트래킹")
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
    parser.add_argument("--no-stable-id", action="store_true",
                        help="화면 이탈 후에도 같은 ID 유지 기능 끄기")
    parser.add_argument("--lost-ttl", type=int, default=900,
                        help="재등장 시 같은 ID를 기억하는 최대 프레임 수")
    parser.add_argument("--hist-match", type=float, default=0.62,
                        help="재식별: 위치 가중 끔일 때만 쓰는 HSV 상관 임계값")
    parser.add_argument("--no-position-reid", action="store_true",
                        help="재식별에서 화면 위치 점수 끄기 (이동 카메라에 유리)")
    parser.add_argument("--position-sigma", type=float, default=200.0,
                        help="위치 점수: 중심 거리 이 정도(px)면 점수 거의 0")
    parser.add_argument("--combined-hist-weight", type=float, default=0.65,
                        help="재식별 결합 시 HSV 비중 (나머지는 위치)")
    parser.add_argument("--combined-match", type=float, default=0.55,
                        help="HSV+위치 결합 점수 임계값 (위치 가중 켤 때)")
    parser.add_argument("--byte-buffer", type=int, default=120,
                        help="ByteTrack lost_track_buffer (프레임)")
    args = parser.parse_args()

    kwargs = dict(
        source=args.source,
        conf=args.conf,
        nms=args.nms,
        compensation=args.compensation,
        min_frames=args.min_frames,
        stable_ids=not args.no_stable_id,
        lost_ttl_frames=args.lost_ttl,
        hist_match_thresh=args.hist_match,
        use_position_reid=not args.no_position_reid,
        position_sigma_px=args.position_sigma,
        combined_hist_weight=args.combined_hist_weight,
        combined_match_thresh=args.combined_match,
        byte_lost_buffer=args.byte_buffer,
    )
    if args.model is not None:
        kwargs["model_path"] = args.model

    run(**kwargs)


if __name__ == "__main__":
    main()
