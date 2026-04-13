#!/usr/bin/env python3
"""토마토 실시간 트래킹 CLI"""

import argparse
import sys
from pathlib import Path

# src를 path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tracker import run, CONFIG


def main():
    parser = argparse.ArgumentParser(description="토마토 실시간 트래킹")
    parser.add_argument("--source", default="0", help="영상 소스 (웹캠=0, 파일=경로)")
    parser.add_argument("--model", default=None, help="YOLO 모델 경로")
    parser.add_argument("--roi-half-width", type=int, default=320, help="ROI 반폭 (픽셀)")
    parser.add_argument("--out", default=None, help="출력 파일 경로")
    parser.add_argument("--headless", action="store_true", help="창 없이 실행")
    parser.add_argument("--debug", action="store_true", help="디버그 출력")
    parser.add_argument("--save-results", default=None, metavar="PATH",
                        help="결과 저장 경로 (예: out/result → result.csv, result.json 생성)")
    
    # CONFIG 오버라이드 옵션
    parser.add_argument("--conf", type=float, default=None, help="검출 신뢰도")
    parser.add_argument("--center-dist", type=int, default=None, help="ROI 내 연속 매칭 최대 거리 (픽셀)")

    args = parser.parse_args()

    # CONFIG 업데이트
    if args.conf:
        CONFIG["conf"] = args.conf
    if args.center_dist:
        CONFIG["center_max_dist"] = args.center_dist
    if args.debug:
        CONFIG["debug"] = True
    
    # 실행
    kwargs = {
        "source": args.source,
        "roi_half_width": args.roi_half_width,
        "output_path": args.out,
        "show_window": not args.headless,
        "debug": args.debug,
        "save_results": args.save_results,
    }
    if args.model:
        kwargs["model_path"] = args.model
    
    run(**kwargs)


if __name__ == "__main__":
    main()
