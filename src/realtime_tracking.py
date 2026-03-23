"""
Real-time object tracking with YOLO26 + ByteTrack.
Supports optional Camera Motion Compensation for moving cameras.

Usage:
    # Fixed camera (webcam)
    python src/realtime_tracking.py

    # Fixed camera (RTSP stream)
    python src/realtime_tracking.py --source rtsp://192.168.0.100:554/stream

    # Moving camera with motion compensation
    python src/realtime_tracking.py --compensation

    # Custom model path and confidence
    python src/realtime_tracking.py --model path/to/best.pt --conf 0.3
"""

import argparse
import time

import cv2
import supervision as sv
from ultralytics import YOLO
from trackers import ByteTrackTracker

COLOR_PALETTE = sv.ColorPalette.from_hex([
    "#ffff00", "#ff9b00", "#ff8080", "#ff66b2", "#ff66ff", "#b266ff",
    "#9999ff", "#3399ff", "#66ffff", "#33ff99", "#66ff66", "#99ff00",
])

DEFAULT_MODEL = r"runs\yolo26_merged_tomato\weights\best.pt"


def build_annotators():
    box = sv.BoxAnnotator(
        color=COLOR_PALETTE,
        color_lookup=sv.ColorLookup.TRACK,
    )
    label = sv.LabelAnnotator(
        color=COLOR_PALETTE,
        color_lookup=sv.ColorLookup.TRACK,
        text_color=sv.Color.BLACK,
        text_scale=0.8,
    )
    trace = sv.TraceAnnotator(
        color=COLOR_PALETTE,
        color_lookup=sv.ColorLookup.TRACK,
        thickness=2,
        trace_length=100,
    )
    return box, label, trace


def build_motion_compensator():
    from trackers import MotionEstimator, MotionAwareTraceAnnotator

    estimator = MotionEstimator(
        max_points=500,
        min_distance=10,
        quality_level=0.001,
        ransac_reproj_threshold=1.0,
    )
    motion_trace = MotionAwareTraceAnnotator(
        color=COLOR_PALETTE,
        color_lookup=sv.ColorLookup.TRACK,
        thickness=2,
        trace_length=100,
    )
    return estimator, motion_trace


def run(args):
    model = YOLO(args.model)
    tracker = ByteTrackTracker(
        minimum_consecutive_frames=args.min_frames,
    )
    box_ann, label_ann, trace_ann = build_annotators()

    motion_estimator = None
    motion_trace_ann = None
    if args.compensation:
        motion_estimator, motion_trace_ann = build_motion_compensator()
        print("[INFO] Camera Motion Compensation ENABLED")
    else:
        print("[INFO] Camera Motion Compensation DISABLED (fixed camera mode)")

    source = args.source
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {args.source}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Source: {args.source} ({w}x{h})")
    print(f"[INFO] Model: {args.model}")
    print(f"[INFO] Confidence: {args.conf} / NMS: {args.nms}")
    print("[INFO] Press 'q' to quit, 'r' to reset tracker, 'c' to toggle compensation")

    fps_avg = 0.0
    alpha = 0.1

    while True:
        t0 = time.perf_counter()

        ret, frame = cap.read()
        if not ret:
            break

        coord_transform = None
        if motion_estimator is not None:
            coord_transform = motion_estimator.update(frame)

        results = model(frame, conf=args.conf, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results).with_nms(
            threshold=args.nms,
        )
        detections = tracker.update(detections)

        annotated = frame.copy()
        annotated = box_ann.annotate(annotated, detections)

        if motion_trace_ann is not None and coord_transform is not None:
            annotated = motion_trace_ann.annotate(
                annotated, detections, coord_transform=coord_transform,
            )
        else:
            annotated = trace_ann.annotate(annotated, detections)

        labels = (
            [str(tid) for tid in detections.tracker_id]
            if detections.tracker_id is not None
            else []
        )
        annotated = label_ann.annotate(annotated, detections, labels)

        dt = time.perf_counter() - t0
        fps = 1.0 / dt if dt > 0 else 0
        fps_avg = alpha * fps + (1 - alpha) * fps_avg if fps_avg > 0 else fps

        n_objects = len(detections)
        cv2.putText(
            annotated,
            f"FPS: {fps_avg:.1f} | Objects: {n_objects}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        cv2.imshow("YOLO26 Real-time Tracking", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            tracker.reset()
            if motion_estimator is not None:
                motion_estimator.reset()
            print("[INFO] Tracker reset")
        elif key == ord("c"):
            if motion_estimator is None:
                motion_estimator, motion_trace_ann = build_motion_compensator()
                print("[INFO] Compensation ON")
            else:
                motion_estimator = None
                motion_trace_ann = None
                print("[INFO] Compensation OFF")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Done.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-time YOLO26 + ByteTrack tracking",
    )
    parser.add_argument(
        "--source", default="0",
        help="Video source: webcam index (0,1,...) or RTSP/file path",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help="Path to YOLO model weights",
    )
    parser.add_argument(
        "--conf", type=float, default=0.2,
        help="Detection confidence threshold",
    )
    parser.add_argument(
        "--nms", type=float, default=0.3,
        help="NMS IoU threshold",
    )
    parser.add_argument(
        "--compensation", action="store_true",
        help="Enable Camera Motion Compensation (for moving cameras)",
    )
    parser.add_argument(
        "--min-frames", type=int, default=1,
        help="Minimum consecutive frames before confirming a track",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
