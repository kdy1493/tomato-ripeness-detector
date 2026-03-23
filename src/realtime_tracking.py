"""Real-time object tracking with YOLO26 + ByteTrack."""

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


def run(*, source="0", model_path=DEFAULT_MODEL, conf=0.2, nms=0.3,
        compensation=False, min_frames=1):
    """Run real-time tracking loop.

    Args:
        source: Webcam index (str/int) or RTSP/file path.
        model_path: Path to YOLO weights.
        conf: Detection confidence threshold.
        nms: NMS IoU threshold.
        compensation: Enable Camera Motion Compensation.
        min_frames: Minimum consecutive frames before confirming a track.
    """
    model = YOLO(model_path)
    tracker = ByteTrackTracker(minimum_consecutive_frames=min_frames)
    box_ann, label_ann, trace_ann = build_annotators()

    motion_estimator = None
    motion_trace_ann = None
    if compensation:
        motion_estimator, motion_trace_ann = build_motion_compensator()
        print("[INFO] Camera Motion Compensation ENABLED")
    else:
        print("[INFO] Camera Motion Compensation DISABLED (fixed camera mode)")

    vid_source = int(source) if isinstance(source, str) and source.isdigit() else source
    cap = cv2.VideoCapture(vid_source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Source: {source} ({w}x{h})")
    print(f"[INFO] Model: {model_path}")
    print(f"[INFO] Confidence: {conf} / NMS: {nms}")
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

        results = model(frame, conf=conf, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results).with_nms(
            threshold=nms,
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


if __name__ == "__main__":
    run()
