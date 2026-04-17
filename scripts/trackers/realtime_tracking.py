#!/usr/bin/env python3
"""토마토 실시간 트래킹"""

import sys
from pathlib import Path

# src를 path에 추가
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import tracker


# ============================================================
# 설정 - 이것만 수정하면 됨!
# ============================================================
CONFIG = {
    # 영상 소스 및 입출력
    "source": "notebook/rgb.mp4",     # 영상 소스 (0=웹캠, 파일 경로)
    "model_path": None,               # YOLO 모델 경로 (None=기본 모델)
    "roi_half_width": 320,            # ROI 반폭 (픽셀)
    "output_path": "notebook/output_bytetrack_new.mp4",              # 출력 영상 파일 경로
    "show_window": True,              # 창 표시 여부
    "save_results": None,             # 결과 저장 경로 (예: "out/result")
    
    # Detection
    "conf": 0.5,              # 검출 신뢰도 (낮춰서 miss 감소)
    "nms": 0.3,               # NMS threshold
    
    # 구조 제약 (토마토 군집 특성)
    "max_y_diff": 100.0,      # 최대 y축 이동 (토마토는 비슷한 높이)
    "max_area_ratio": 3.0,    # 최대 면적 변화 비율
    "max_backward_x": 50.0,   # 주 이동 방향의 반대쪽으로 허용할 최대 이동량
    "max_movement_unknown": 300,  # 방향 불명확 시 최대 이동 (초기/방향 전환 시 복구 허용)
    
    # 방향 감지 (동적 좌→우 / 우→좌 판단)
    "direction_min_tracks": 3,     # 방향 감지 최소 트랙 수
    "direction_dx_threshold": 5.0,  # 방향 판정 dx 임계값 (픽셀)
    "direction_ema_alpha": 0.3,     # EMA 가중치 (0~1, 클수록 최근 값 반영)
    "direction_hysteresis": 2.0,    # 방향 전환 히스테리시스 (출렁임 방지)
    "count_unknown_ema_threshold": 3.0,  # 방향 불명확 시 카운트할 최소 EMA 절댓값
    
    # Suspicious New ID 복구
    "suspicious_new_match_dist": 350,  # 출구 쪽 새 ID를 lost에서 복구할 최대 거리 (큰 위치 변화 대응)
    "suspicious_recover_reid_threshold": 0.35,  # 복구 시 최소 ReID 유사도 (거리 넓혀서 ReID 조금 강화)
    "suspicious_lost_frames_penalty": 2.0,  # lost_frames당 비용 패널티

    # ByteTrack (클래스별 독립 sv.ByteTrack)
    "byte_track_activation_threshold": 0.25,  # 트랙 활성화 최소 신뢰도
    "byte_minimum_matching_threshold": 0.8,   # IoU 매칭 임계값 (높을수록 ID switching 감소)
    "byte_buffer": 30,        # ByteTrack lost 트랙 유지 프레임 수

    # Stable ID Lost Buffer (적응적)
    "lost_buffer_frames": 20,      # 카운트 완료된 트랙의 lost 버퍼 (짧게)
    "lost_buffer_uncounted": 150,  # 카운트 전 트랙의 lost 버퍼 (길게, 장기 가려짐 대응)

    # ROI 내 연속 프레임 ID 유지 (픽셀)
    "center_max_dist": 200,   # 연속 프레임 중심점 최대 거리 (500→200으로 축소)

    # 카운팅 (입구 라인)
    "counting_entry_offset": 50,   # ROI 입구 경계로부터의 오프셋 (픽셀)
    "counting_min_consecutive": 3, # 카운트 확정에 필요한 최소 연속 검출 프레임 수

    # ReID (Re-Identification)
    "use_reid": True,         # ReID 특징 사용 여부
    "reid_weight": 0.3,       # ReID 특징 가중치 (0~1, 나머지는 거리 가중치)
    "reid_threshold": 0.5,    # ReID 유사도 임계값 (0.3→0.5로 상향, 실제 로직과 일치)
    "reid_hist_bins": 32,     # 색상 히스토그램 빈 개수

    # 시각화
    "show_trace": False,      # 궤적 표시 여부

    # 카메라 움직임 보정 (레일 카메라 필수)
    # ID 매칭 시 이전 트랙 좌표를 현재 프레임으로 워핑하여 카메라 이동분 상쇄
    "motion_compensation": True,
    "motion_max_points": 500,
    "motion_min_distance": 10,
    "motion_block_size": 3,
    "motion_quality_level": 0.001,
    "motion_ransac_reproj_threshold": 1.0,
    "trace_length": 80,

    # 군집 내 순서 제약 (실내 환경 전용)
    # 같은 클래스의 토마토 x좌표 순서가 프레임 간 보존되는지 검증
    # 바람 없는 실내에서 물리적 순서는 불변 → ID swap 방지
    "use_order_constraint": True,

    # 디버그
    "debug": True,
}
# ============================================================

# CONFIG를 tracker 모듈에 주입
tracker.CONFIG = CONFIG


def main():
    """기본 실행"""
    tracker.run(
        source=CONFIG["source"],
        model_path=CONFIG["model_path"],
        roi_half_width=CONFIG["roi_half_width"],
        output_path=CONFIG["output_path"],
        show_window=CONFIG["show_window"],
        save_results=CONFIG["save_results"],
    )


if __name__ == "__main__":
    main()
