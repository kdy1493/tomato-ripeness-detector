#!/usr/bin/env python3
"""
트래커 벤치마크: SORT / ByteTrack / DeepSORT / tracker 비교

각 basic 스크립트의 run() 함수를 직접 import해서 실행합니다.
  - basic_bytetracker.run(config)
  - basic_sort.run(config)
  - basic_deepsort.run(config)
  - tracker.run_benchmark(config)   # src/tracker.py

출력 (output_dir 기준):
  mot/bytetrack.txt      ─ MOT 형식 결과 (frame, id, x, y, w, h, conf, class)
  mot/sort.txt
  mot/deepsort.txt
  mot/tracker.txt
  videos/bytetrack.mp4   ─ 결과 영상 (save_video=True 일 때)
  ...
  summary.csv            ─ 지표 요약
  comparison.png         ─ 비교 차트
  mot/gt.txt             ─ --gt 사용 시 GT 복사 (표에서 GT 행과 동일)

GT 없이 실행:
  python scripts/benchmark.py

GT 있으면 MOTA·IDF1·IDSW 추가 계산:
  python scripts/benchmark.py --gt benchmark/gt.txt

특정 tracker만 실행:
  python scripts/benchmark.py --trackers bytetrack,tracker

설정: CONFIG_SHARED 로 영상·검출·ROI만 통일, 트래커별 권장값은 각 basic_*.py CONFIG 와 동일.

GT 파일 형식 (MOT Challenge):
  frame_id, track_id, x, y, w, h, conf, class_id
  (1-based frame_id, bbox는 x_topleft·y_topleft·width·height)
"""

import argparse
import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(SCRIPTS_DIR))


def resolve_gt_path(gt: str) -> Path:
    """cwd 또는 레포 루트 기준 GT MOT 파일 경로."""
    p = Path(gt)
    if p.is_file():
        return p
    alt = REPO_ROOT / gt
    if alt.is_file():
        return alt
    raise FileNotFoundError(f"GT 없음: {gt!r} (레포 기준 시도: {alt})")

# ============================================================
# 설정
#   CONFIG_SHARED     — 영상·모델·검출·ROI (모든 트래커 동일 입력)
#   *_RECOMMENDED     — basic_bytetracker / basic_sort / basic_deepsort / tracker.py 권장값
# ============================================================

CONFIG_SHARED = {
    "source":     "notebook/rgb.mp4",
    "model_path": "runs/yolo26_custom_tomato/trained_yolo26_custom.pt",
    "output_dir": "benchmark",
    "save_video": True,
    "conf":       0.5,
    "iou":        0.3,
    "roi_half_width": 320,
}

# basic_bytetracker.py CONFIG (ByteTrack 블록)
BYTETRACK_RECOMMENDED = {
    "track_activation_threshold": 0.25,
    "lost_track_buffer":          30,
    "minimum_matching_threshold": 0.8,
    "frame_rate":                 30,
}

# basic_sort.py CONFIG (SORT 블록) — minimum_iou_threshold 는 SORT 전용 (ByteTrack 의 matching 과 다름)
SORT_RECOMMENDED = {
    "track_activation_threshold": 0.25,
    "lost_track_buffer":          30,
    "minimum_iou_threshold":        0.3,
    "minimum_consecutive_frames": 1,
    "frame_rate":                 30,
}

# basic_deepsort.py CONFIG (DeepSORT 블록)
DEEPSORT_RECOMMENDED = {
    "max_age":         30,
    "n_init":           3,
    "max_cosine_dist":  0.3,
    "nn_budget":       100,
}

# src/tracker.py run_benchmark 가 읽는 키 (프로젝트 기본 CONFIG 와 동일)
TRACKER_RECOMMENDED = {
    "byte_track_activation_threshold": 0.25,
    "byte_minimum_matching_threshold": 0.8,
    "byte_lost_track_buffer":          30,
    "tnew_roi_half_width":    320,
    "tnew_max_y_diff":        100.0,
    "tnew_max_area_ratio":    3.0,
    "tnew_max_backward_x":    50.0,
    "tnew_max_movement_unknown": 300,
    "tnew_center_max_dist":   200,
    "tnew_use_reid":          True,
    "tnew_reid_weight":       0.3,
    "tnew_reid_threshold":    0.5,
    "tnew_reid_hist_bins":    32,
    "tnew_lost_buffer_frames":    20,
    "tnew_lost_buffer_uncounted": 150,
    "tnew_use_order_constraint":  True,
    "tnew_motion_compensation":   True,
    "tnew_motion_max_points":     500,
    "tnew_motion_min_distance":   10,
    "tnew_motion_block_size":     3,
    "tnew_motion_quality_level":  0.001,
    "tnew_motion_ransac_reproj_threshold": 1.0,
    "tnew_direction_min_tracks":  3,
    "tnew_direction_dx_threshold": 5.0,
    "tnew_direction_ema_alpha":   0.3,
    "tnew_direction_hysteresis":  2.0,
    "tnew_suspicious_new_match_dist":      350,
    "tnew_suspicious_recover_reid_threshold": 0.35,
    "tnew_suspicious_lost_frames_penalty": 2.0,
    "tnew_count_unknown_ema_threshold":    3.0,
    "tnew_counting_entry_offset":          50,
    "tnew_counting_min_consecutive":       3,
    "tnew_trace_length":                   80,
}

# main()·경로용 (output_dir, save_video)
CONFIG = CONFIG_SHARED
# ============================================================

CLASS_NAMES = {0: "ripe", 1: "unripe"}


# ---------------------------------------------------------------------------
# 데이터 구조
# ---------------------------------------------------------------------------

@dataclass
class TrackerResult:
    name: str
    mot_rows: List[Tuple] = field(default_factory=list)
    fps_avg: float = 0.0
    total_frames: int = 0
    unique_ids: Dict[int, set] = field(default_factory=lambda: {0: set(), 1: set()})


def _from_run_result(name: str, result: dict) -> TrackerResult:
    """basic 스크립트 run() 반환값 → TrackerResult 변환"""
    return TrackerResult(
        name=name,
        mot_rows=result["mot_rows"],
        fps_avg=result["fps_avg"],
        total_frames=result["total_frames"],
        unique_ids=result["unique_ids"],
    )


def load_gt_tracker_result(gt_path: Path) -> TrackerResult:
    """MOT CSV GT → TrackerResult (벤치마크 표·지표용)."""
    path = gt_path
    if not path.is_file():
        raise FileNotFoundError(f"GT 없음: {path}")

    mot_rows: List[Tuple] = []
    unique_ids: Dict[int, set] = {0: set(), 1: set()}
    max_frame = 0

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            raise ValueError(f"빈 GT: {path}")
        for row in reader:
            if len(row) < 8:
                continue
            fid = int(row[0])
            tid = int(row[1])
            x, y, w, h = float(row[2]), float(row[3]), float(row[4]), float(row[5])
            conf = float(row[6]) if row[6] not in ("",) else 1.0
            cid = int(row[7])
            max_frame = max(max_frame, fid)
            mot_rows.append((fid, tid, x, y, w, h, conf, cid))
            if cid in unique_ids:
                unique_ids[cid].add(tid)

    return TrackerResult(
        name="GT",
        mot_rows=mot_rows,
        fps_avg=0.0,
        total_frames=max_frame,
        unique_ids=unique_ids,
    )


# ---------------------------------------------------------------------------
# 각 tracker 실행 (basic 스크립트 run() 호출)
# ---------------------------------------------------------------------------

def _bytetrack_config(video_path: Optional[Path]) -> dict:
    return {
        **CONFIG_SHARED,
        **BYTETRACK_RECOMMENDED,
        "output_path": str(video_path) if video_path else None,
        "show_window": False,
        "show_trace":  False,
    }


def _sort_config(video_path: Optional[Path]) -> dict:
    return {
        **CONFIG_SHARED,
        **SORT_RECOMMENDED,
        "output_path": str(video_path) if video_path else None,
        "show_window": False,
        "show_trace":  False,
    }


def _deepsort_config(video_path: Optional[Path]) -> dict:
    return {
        **CONFIG_SHARED,
        **DEEPSORT_RECOMMENDED,
        "output_path": str(video_path) if video_path else None,
        "show_window": False,
        "show_trace":  False,
    }


def _tracker_config(video_path: Optional[Path]) -> dict:
    return {
        **CONFIG_SHARED,
        **TRACKER_RECOMMENDED,
        "output_path": str(video_path) if video_path else None,
    }


# ---------------------------------------------------------------------------
# MOT 파일 저장
# ---------------------------------------------------------------------------

def save_mot(result: TrackerResult, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_id", "track_id", "x", "y", "w", "h", "conf", "class_id"])
        for row in sorted(result.mot_rows, key=lambda r: (r[0], r[7], r[1])):
            writer.writerow(row)
    print(f"  → MOT 저장: {path}")


# ---------------------------------------------------------------------------
# Proxy 지표 (GT 없이)
# ---------------------------------------------------------------------------

def proxy_metrics(result: TrackerResult) -> dict:
    total_ids  = len(result.unique_ids[0]) + len(result.unique_ids[1])
    avg_dets   = len(result.mot_rows) / result.total_frames if result.total_frames else 0
    instability = total_ids / avg_dets if avg_dets > 0 else float("inf")
    return {
        "ripe_ids":       len(result.unique_ids[0]),
        "unripe_ids":     len(result.unique_ids[1]),
        "total_ids":      total_ids,
        "fps":            round(result.fps_avg, 1),
        "avg_dets":       round(avg_dets, 1),
        "id_instability": round(instability, 2),
    }


# ---------------------------------------------------------------------------
# MOT 지표 (GT 있을 때)
# ---------------------------------------------------------------------------

def _np_asfarray_shim() -> None:
    """NumPy 2.0에서 제거된 np.asfarray — motmetrics.distances.iou_matrix 호환."""
    if hasattr(np, "asfarray"):
        return

    def _asfarray(a, dtype=np.float64):
        return np.asarray(a, dtype=dtype)

    np.asfarray = _asfarray  # type: ignore[method-assign]


def mot_metrics(result: TrackerResult, gt_path: str) -> Optional[dict]:
    try:
        import motmetrics as mm
    except ImportError:
        print("[WARN] motmetrics 미설치 → pip install motmetrics")
        return None

    _np_asfarray_shim()

    def _load(path):
        data: Dict[int, Dict[int, list]] = {}
        with open(path) as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if len(row) < 6:
                    continue
                fid, tid = int(row[0]), int(row[1])
                data.setdefault(fid, {})[tid] = [float(row[2]), float(row[3]),
                                                  float(row[4]), float(row[5])]
        return data

    gt_data  = _load(gt_path)
    hyp_data: Dict[int, Dict[int, list]] = {}
    for row in result.mot_rows:
        hyp_data.setdefault(row[0], {})[row[1]] = [row[2], row[3], row[4], row[5]]

    acc = mm.MOTAccumulator(auto_id=True)
    for fid in sorted(set(list(gt_data) + list(hyp_data))):
        gt_f  = gt_data.get(fid, {})
        hyp_f = hyp_data.get(fid, {})
        gt_ids, hyp_ids = list(gt_f), list(hyp_f)
        if not gt_ids and not hyp_ids:
            continue
        dist = (mm.distances.iou_matrix([gt_f[i] for i in gt_ids],
                                        [hyp_f[i] for i in hyp_ids], max_iou=0.5)
                if gt_ids and hyp_ids else np.empty((len(gt_ids), len(hyp_ids))))
        acc.update(gt_ids, hyp_ids, dist)

    mh  = mm.metrics.create()
    row = mh.compute(
        acc,
        metrics=["num_switches", "mota", "idf1", "recall",
                 "precision", "num_false_positives", "num_misses"],
        name=result.name,
    ).iloc[0]

    return {
        "MOTA":  round(float(row["mota"]) * 100, 1),
        "IDF1":  round(float(row["idf1"]) * 100, 1),
        "IDSW":  int(row["num_switches"]),
        "Recall": round(float(row["recall"]) * 100, 1),
        "Prec":  round(float(row["precision"]) * 100, 1),
        "FP":    int(row["num_false_positives"]),
        "FN":    int(row["num_misses"]),
    }


# ---------------------------------------------------------------------------
# 결과 출력 및 차트
# ---------------------------------------------------------------------------

def print_table(results: List[TrackerResult], metrics_list: List[dict]):
    print("\n" + "=" * 80)
    has_mot = any("MOTA" in m for m in metrics_list)
    print(f"{'Tracker':<16} {'FPS':>6} {'ripe IDs':>9} {'unripe IDs':>11} "
          f"{'total IDs':>10} {'ID instab.':>11}", end="")
    if has_mot:
        print(f" {'MOTA':>7} {'IDF1':>7} {'IDSW':>6}", end="")
    print()
    print("-" * 80)
    for r, m in zip(results, metrics_list):
        if isinstance(m.get("fps"), str):
            fps_s = f"{m['fps']:>6}"
        else:
            fps_s = f"{float(m['fps']):>6.1f}"
        line = (f"{r.name:<16} {fps_s} {m['ripe_ids']:>9} {m['unripe_ids']:>11} "
                f"{m['total_ids']:>10} {m['id_instability']:>11}")
        if has_mot:
            mota = m.get("MOTA", "-")
            idf1 = m.get("IDF1", "-")
            idsw = m.get("IDSW", "-")
            line += (f" {str(mota):>7} {str(idf1):>7} {str(idsw):>6}")
        print(line)
    print("=" * 80)
    print("※ ID instability = 총 unique IDs / 프레임당 평균 검출 수  (낮을수록 ID switching 적음)")
    if has_mot:
        print("※ GT 행: MOTA·IDF1·IDSW 는 GT 대 GT (이론상 완전 일치 시 상한선)")


def save_summary_csv(results, metrics_list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    seen = set()
    for m in metrics_list:
        for k in m:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["tracker"] + keys)
        for r, m in zip(results, metrics_list):
            writer.writerow([r.name] + [m.get(k, "") for k in keys])
    print(f"\n[저장] {path}")


def plot_comparison(results: List[TrackerResult], metrics_list: List[dict], out_path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib 미설치 → pip install matplotlib")
        return

    has_mot = any("MOTA" in m for m in metrics_list)
    ncols   = 4 if has_mot else 3
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))
    fig.suptitle("Tracker Comparison", fontsize=14, fontweight="bold")
    names   = [r.name for r in results]
    nt_idx  = [i for i, r in enumerate(results) if r.name != "GT"]
    palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#937860", "#8C8C8C"]

    def _bar(ax, values, title, ylabel, lower_better=False, bar_names=None):
        bn = bar_names if bar_names is not None else names
        if not values:
            ax.set_title(title)
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            return
        bars = ax.bar(bn, values, color=palette[: len(bn)], width=0.5)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        vmax = max(values)
        ax.set_ylim(0, vmax * 1.25 if vmax > 0 else 1)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02 * max(values),
                    f"{val}", ha="center", va="bottom", fontsize=10)
        best = (min if lower_better else max)(range(len(values)), key=lambda i: values[i])
        bars[best].set_edgecolor("gold")
        bars[best].set_linewidth(3)
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y", alpha=0.3)

    _bar(axes[0], [m["total_ids"] for m in metrics_list],
         "Total Unique IDs\n(↓ fewer = less ID switch)", "# IDs", lower_better=True)
    _bar(axes[1], [m["id_instability"] for m in metrics_list],
         "ID Instability\n(↓ lower = more stable)", "score", lower_better=True)
    fps_vals = [metrics_list[i]["fps"] for i in nt_idx
                if isinstance(metrics_list[i].get("fps"), (int, float))]
    fps_names = [results[i].name for i in nt_idx
                 if isinstance(metrics_list[i].get("fps"), (int, float))]
    _bar(axes[2], fps_vals,
         "Speed (FPS)\n(↑ higher = faster)", "FPS", lower_better=False,
         bar_names=fps_names)
    if has_mot:
        mota_vals = [metrics_list[i].get("MOTA", 0) for i in nt_idx]
        mota_names = [results[i].name for i in nt_idx]
        _bar(axes[3], mota_vals,
             "MOTA (%)\n(↑ higher = better)", "%", lower_better=False,
             bar_names=mota_names)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    print(f"[저장] {out_path}")
    plt.close()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Tracker benchmark")
    parser.add_argument("--gt", type=str, default=None,
                        help="GT MOT 파일 경로 (없으면 proxy 지표만)")
    parser.add_argument("--trackers", type=str,
                        default="bytetrack,sort,deepsort,tracker",
                        help="실행할 tracker 목록 (comma-separated)")
    args = parser.parse_args()

    out_dir   = REPO_ROOT / CONFIG["output_dir"]
    mot_dir   = out_dir / "mot"
    video_dir = out_dir / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)

    to_run  = [t.strip() for t in args.trackers.split(",")]
    results: List[TrackerResult] = []

    # ── ByteTrack ────────────────────────────────────────────
    if "bytetrack" in to_run:
        import basic_bytetracker
        vp = video_dir / "bytetrack.mp4" if CONFIG["save_video"] else None
        r  = _from_run_result("ByteTrack",
                               basic_bytetracker.run(_bytetrack_config(vp)))
        save_mot(r, mot_dir / "bytetrack.txt")
        results.append(r)

    # ── SORT ─────────────────────────────────────────────────
    if "sort" in to_run:
        import basic_sort
        vp = video_dir / "sort.mp4" if CONFIG["save_video"] else None
        r  = _from_run_result("SORT",
                               basic_sort.run(_sort_config(vp)))
        save_mot(r, mot_dir / "sort.txt")
        results.append(r)

    # ── DeepSORT ─────────────────────────────────────────────
    if "deepsort" in to_run:
        import basic_deepsort
        vp = video_dir / "deepsort.mp4" if CONFIG["save_video"] else None
        r  = _from_run_result("DeepSORT",
                               basic_deepsort.run(_deepsort_config(vp)))
        save_mot(r, mot_dir / "deepsort.txt")
        results.append(r)

    # ── tracker (src/tracker.py) ─────────────────────────────
    if "tracker" in to_run:
        import tracker
        vp = video_dir / "tracker.mp4" if CONFIG["save_video"] else None
        r  = _from_run_result("tracker",
                               tracker.run_benchmark(_tracker_config(vp)))
        save_mot(r, mot_dir / "tracker.txt")
        results.append(r)

    if not results:
        print("[ERROR] 실행된 tracker가 없습니다.")
        return

    # ── GT 행 (--gt): 표·CSV·차트 첫 줄 + mot/gt.txt ──────────
    gt_path: Optional[Path] = None
    if args.gt:
        try:
            gt_path = resolve_gt_path(args.gt)
        except FileNotFoundError as e:
            print(f"[ERROR] {e}")
            return
        try:
            gt_res = load_gt_tracker_result(gt_path)
        except ValueError as e:
            print(f"[ERROR] GT 로드 실패: {e}")
            return
        results.insert(0, gt_res)
        save_mot(gt_res, mot_dir / "gt.txt")

    # ── 지표 계산 ────────────────────────────────────────────
    metrics_list = []
    gt_str = str(gt_path) if gt_path else None
    for r in results:
        m = proxy_metrics(r)
        if r.name == "GT":
            m["fps"] = "-"
        if gt_str:
            mot_m = mot_metrics(r, gt_str)
            if mot_m:
                m.update(mot_m)
        metrics_list.append(m)

    # ── 출력 ─────────────────────────────────────────────────
    print_table(results, metrics_list)
    save_summary_csv(results, metrics_list, out_dir / "summary.csv")
    plot_comparison(results, metrics_list, out_dir / "comparison.png")
    print(f"\n[완료] 결과: {out_dir}")


if __name__ == "__main__":
    main()
