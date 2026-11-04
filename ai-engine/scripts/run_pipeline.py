r"""Run the R-SAFE AI engine on a video file, a phone IP Webcam stream, or a camera.

Every flagged motorcycle becomes a folder under the configured output directory
(status "pending_review": a human must verify it).

Usage (from ai-engine\):
  .venv\Scripts\python scripts\run_pipeline.py --source clip.mp4 --show
  .venv\Scripts\python scripts\run_pipeline.py --source http://192.168.1.5:8080/video
  .venv\Scripts\python scripts\run_pipeline.py --source 0 --max-frames 900

With --show, press q in the video window to stop.

For a recorded file, detected_at = --start-time (when filming began; default: now)
plus the frame's position in the video. Give --start-time so records show when the
riders were really filmed, e.g. --start-time 2026-11-06T07:30:00 (Philippine time).
"""

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rsafe.config import ConfigError, load_config  # noqa: E402
from rsafe.app.session import VideoRecorder  # noqa: E402
from rsafe.pipeline import Engine  # noqa: E402
from rsafe.video.source import VideoSource, frame_timestamp, video_fps  # noqa: E402

AI_ENGINE_DIR = Path(__file__).resolve().parents[1]
WINDOW = "R-SAFE (q to quit)"
WINDOW_SIZE = (1280, 720)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", help="video file, http(s)/rtsp URL, or camera index (default: video.source in config)")
    parser.add_argument("--config", type=Path, default=AI_ENGINE_DIR / "configs" / "default.yaml")
    parser.add_argument("--show", action="store_true", help="show the annotated video")
    parser.add_argument("--max-frames", type=int, help="stop after this many source frames")
    parser.add_argument("--start-time", type=datetime.fromisoformat,
                        help="when a recorded file started, e.g. 2026-11-06T07:30:00 (local time)")
    parser.add_argument("--save-video", type=Path,
                        help="also save the analysed frames with boxes as an .mp4 (video files only)")
    args = parser.parse_args()

    try:
        cfg = load_config(args.config)
        source_text = args.source or cfg.video.source
        if not source_text:
            raise ConfigError("No video source: pass --source or set video.source in the config")
        source = VideoSource(source_text, cfg.video.camera_id)
    except (ConfigError, FileNotFoundError) as error:
        print(f"[R-SAFE] {error}", file=sys.stderr)
        return 2

    logging.basicConfig(level=cfg.logging.level, format="[R-SAFE] %(levelname)s %(name)s: %(message)s")
    print("[R-SAFE] Loading models...")
    engine = Engine(cfg)
    print(f"[R-SAFE] {engine.performance.describe()}")
    cfg = engine.cfg  # settings for this computer's performance level
    pipeline = engine.new_pipeline()
    print(f"[R-SAFE] Reading {source.info.kind}: {source.info.uri}")
    if args.show:
        import cv2

        # resizable window that fits a laptop screen; drag its corner to change the size
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, *WINDOW_SIZE)

    is_file = source.info.kind == "file"
    file_start = args.start_time or datetime.now(timezone.utc)
    fps = video_fps(source.info.uri) if is_file else 0.0
    recorder = None
    if args.save_video:
        if is_file and fps > 0:
            # one frame per analysed frame, so the saved video plays at the original speed
            recorder = VideoRecorder(args.save_video, fps / cfg.video.frame_stride)
        else:
            print("[R-SAFE] --save-video works for video files only; ignored", file=sys.stderr)
    processed = 0
    written = []
    started = time.perf_counter()
    try:
        for index, frame in source.frames():
            if args.max_frames is not None and index >= args.max_frames:
                break
            if index % cfg.video.frame_stride:
                continue
            now = frame_timestamp(file_start, index, fps) if is_file else datetime.now(timezone.utc)
            written += pipeline.process_frame(index, frame, source.info, now)
            processed += 1
            if recorder:
                recorder.write(pipeline.annotate(frame))
            if args.show:
                import cv2

                cv2.imshow(WINDOW, pipeline.annotate(frame))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        print("[R-SAFE] Stopped by user")
    finally:
        source.close()
        written += pipeline.finish(source.info)
        if recorder:
            recorder.close()
        if args.show:
            import cv2

            cv2.destroyAllWindows()

    elapsed = time.perf_counter() - started
    print("[R-SAFE] Summary")
    print(f"  frames processed : {processed} (every {cfg.video.frame_stride} source frames)")
    print(f"  average speed    : {processed / elapsed if elapsed else 0:.1f} processed frames/s")
    print(f"  violations saved : {len(written)} (pending_review)")
    print(f"  output folder    : {cfg.evidence.output_dir}")
    if recorder and recorder.path:
        print(f"  analysed video   : {recorder.path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
