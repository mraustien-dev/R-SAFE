import sys
import cv2
import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from pathlib import Path
from PIL import Image, ImageTk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rsafe.detectors.helmet import HelmetDetector

import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.getLogger("rsafe").setLevel(logging.DEBUG)


# ============================================================
# R-SAFE CONFIGURATION
# ============================================================

CAMERA_URL = "http://127.0.0.1:7000/video"

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "helmet"
    / "best.onnx"
)

CONFIDENCE_THRESHOLD = 0.20
IOU_THRESHOLD = 0.45

DEFAULT_INFERENCE_SIZE = 416
DEFAULT_INFERENCE_INTERVAL = 3

WINDOW_GEOMETRY = "1100x760"
MIN_WINDOW_SIZE = (760, 540)
DISPLAY_REFRESH_MS = 30


class RSAFELiveDetectionGUI:
    """Resizable R-SAFE GUI for IP webcam and video-file inference."""

    def __init__(self, root):
        self.root = root
        self.root.title("R-SAFE | Live Helmet Detection")
        self.root.geometry(WINDOW_GEOMETRY)
        self.root.minsize(*MIN_WINDOW_SIZE)
        self.root.configure(bg="#101820")

        self.running = True
        self.source_running = False
        self.paused = False
        self.source_kind = "none"
        self.source_name = "No source selected"

        self.capture = None
        self.capture_lock = threading.Lock()
        self.frame_lock = threading.Lock()
        self.detection_lock = threading.Lock()

        self.latest_frame = None
        self.latest_detections = []
        self.frame_count = 0
        self.last_inference_frame = 0

        self.camera_fps = 0.0
        self.inference_fps = 0.0
        self.inference_time_ms = 0.0
        self.helmet_count = 0
        self.no_helmet_count = 0

        self.inference_size = DEFAULT_INFERENCE_SIZE
        self.inference_interval = DEFAULT_INFERENCE_INTERVAL

        self.reader_thread = None
        self.inference_thread = None

        if not MODEL_PATH.exists():
            messagebox.showerror("Model Not Found", f"ONNX model not found:\n{MODEL_PATH}")
            self.root.destroy()
            return

        try:
            self.detector = HelmetDetector(
                model_path=str(MODEL_PATH),
                confidence_threshold=CONFIDENCE_THRESHOLD,
                iou_threshold=IOU_THRESHOLD,
                image_size=self.inference_size,
            )
        except Exception as error:
            messagebox.showerror("AI Model Error", str(error))
            self.root.destroy()
            return

        self._build_interface()
        self.inference_thread = threading.Thread(
            target=self.inference_loop, daemon=True
        )
        self.inference_thread.start()

        self.root.protocol("WM_DELETE_WINDOW", self.close_application)
        self.root.after(DISPLAY_REFRESH_MS, self.update_display)

    # ========================================================
    # INTERFACE
    # ========================================================

    def _build_interface(self):
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(2, weight=1)

        title = tk.Label(
            self.root,
            text="R-SAFE LIVE HELMET DETECTION",
            font=("Segoe UI", 20, "bold"),
            fg="#00D4FF",
            bg="#101820",
        )
        title.grid(row=0, column=0, sticky="ew", padx=12, pady=(10, 8))

        controls = tk.Frame(self.root, bg="#1B2733", padx=10, pady=8)
        controls.grid(row=1, column=0, sticky="ew", padx=12, pady=4)

        controls.grid_columnconfigure(8, weight=1)

        tk.Button(
            controls, text="Open Traffic Video", command=self.open_video,
            bg="#00A6CC", fg="white", font=("Segoe UI", 9, "bold")
        ).grid(row=0, column=0, padx=4, pady=3)

        tk.Button(
            controls, text="Connect IP Camera", command=self.open_camera,
            bg="#2D7D46", fg="white", font=("Segoe UI", 9, "bold")
        ).grid(row=0, column=1, padx=4, pady=3)

        self.play_pause_button = tk.Button(
            controls, text="Pause", command=self.toggle_pause,
            bg="#D69E00", fg="white", font=("Segoe UI", 9, "bold")
        )
        self.play_pause_button.grid(row=0, column=2, padx=4, pady=3)

        tk.Button(
            controls, text="Stop Source", command=self.stop_source,
            bg="#B83A3A", fg="white", font=("Segoe UI", 9, "bold")
        ).grid(row=0, column=3, padx=4, pady=3)

        tk.Label(
            controls, text="AI Resolution:", fg="white", bg="#1B2733",
            font=("Segoe UI", 9, "bold")
        ).grid(row=0, column=4, padx=(12, 3))

        self.resolution_var = tk.StringVar(value=str(DEFAULT_INFERENCE_SIZE))
        self.resolution_dropdown = ttk.Combobox(
            controls, textvariable=self.resolution_var,
            values=["416", "512", "640"], state="readonly", width=6
        )
        self.resolution_dropdown.grid(row=0, column=5, padx=3)
        self.resolution_dropdown.bind(
            "<<ComboboxSelected>>", self.change_resolution
        )

        tk.Label(
            controls, text="Interval:", fg="white", bg="#1B2733",
            font=("Segoe UI", 9, "bold")
        ).grid(row=0, column=6, padx=(8, 3))

        self.interval_var = tk.StringVar(value=str(DEFAULT_INFERENCE_INTERVAL))
        self.interval_dropdown = ttk.Combobox(
            controls, textvariable=self.interval_var,
            values=["1", "3", "5"], state="readonly", width=5
        )
        self.interval_dropdown.grid(row=0, column=7, padx=3)
        self.interval_dropdown.bind(
            "<<ComboboxSelected>>", self.change_interval
        )

        tk.Label(
            controls, text="Video Rotation:", fg="white", bg="#1B2733",
            font=("Segoe UI", 9, "bold")
        ).grid(row=0, column=8, padx=(10, 3))

        self.rotation_var = tk.StringVar(value="0°")
        self.rotation_dropdown = ttk.Combobox(
            controls,
            textvariable=self.rotation_var,
            values=["0°", "90° Clockwise", "90° Counterclockwise", "180°"],
            state="readonly",
            width=20,
        )
        self.rotation_dropdown.grid(row=0, column=9, padx=3)
        self.rotation_dropdown.bind(
            "<<ComboboxSelected>>", self.change_rotation
        )

        # The video area expands and contracts with the window.
        video_frame = tk.Frame(self.root, bg="#000000")
        video_frame.grid(row=2, column=0, sticky="nsew", padx=12, pady=8)
        video_frame.grid_columnconfigure(0, weight=1)
        video_frame.grid_rowconfigure(0, weight=1)

        self.video_label = tk.Label(
            video_frame, bg="black", text="Open a traffic video or connect camera",
            fg="#AAB7C4", font=("Segoe UI", 12)
        )
        self.video_label.grid(row=0, column=0, sticky="nsew")

        self.status_label = tk.Label(
            self.root, text="Source: No source selected",
            fg="#AAB7C4", bg="#101820", font=("Segoe UI", 9)
        )
        self.status_label.grid(row=3, column=0, sticky="ew", padx=14, pady=(0, 2))

        performance = tk.Frame(self.root, bg="#1B2733", padx=10, pady=8)
        performance.grid(row=4, column=0, sticky="ew", padx=12, pady=4)

        self.camera_fps_label = self._metric_label(performance, "Camera FPS: 0.0", "#00D4FF")
        self.ai_fps_label = self._metric_label(performance, "AI FPS: 0.00", "#00FF88")
        self.inference_time_label = self._metric_label(performance, "Inference: 0.0 ms", "#FFD700")
        self.helmet_label = self._metric_label(performance, "Helmet: 0", "#00FF88")
        self.no_helmet_label = self._metric_label(performance, "No Helmet: 0", "#FF5555")

    @staticmethod
    def _metric_label(parent, text, color):
        label = tk.Label(
            parent, text=text, fg=color, bg="#1B2733",
            font=("Segoe UI", 10, "bold")
        )
        label.pack(side="left", padx=10)
        return label

    # ========================================================
    # SOURCE CONTROL
    # ========================================================

    def _set_source(self, source, kind, name):
        self.stop_source()

        if source is None or not source.isOpened():
            if source is not None:
                source.release()
            messagebox.showerror("Source Error", f"Could not open:\n{name}")
            return

        with self.capture_lock:
            self.capture = source

        self.source_kind = kind
        self.source_name = name
        self.source_running = True
        self.paused = False
        self.play_pause_button.configure(text="Pause")
        self.status_label.configure(text=f"Source: {name}")

        self.reader_thread = threading.Thread(
            target=self.reader_loop, daemon=True
        )
        self.reader_thread.start()

    def open_video(self):
        path = filedialog.askopenfilename(
            title="Select traffic video",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.m4v"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        cap = cv2.VideoCapture(path)
        self._set_source(cap, "video", Path(path).name)

    def open_camera(self):
        cap = cv2.VideoCapture(CAMERA_URL)
        self._set_source(cap, "camera", CAMERA_URL)

    def stop_source(self):
        self.source_running = False
        self.paused = False

        with self.capture_lock:
            cap = self.capture
            self.capture = None

        if cap is not None:
            cap.release()

        with self.frame_lock:
            self.latest_frame = None
            self.frame_count = 0
            self.last_inference_frame = 0

        with self.detection_lock:
            self.latest_detections = []
            self.helmet_count = 0
            self.no_helmet_count = 0

        self.source_kind = "none"
        self.source_name = "No source selected"
        self.status_label.configure(text="Source: No source selected")
        self.play_pause_button.configure(text="Pause")

    def toggle_pause(self):
        if not self.source_running or self.source_kind != "video":
            return

        self.paused = not self.paused
        self.play_pause_button.configure(
            text="Resume" if self.paused else "Pause"
        )

    # ========================================================
    # READER THREAD
    # ========================================================

    def reader_loop(self):
        fps_counter = 0
        fps_start = time.perf_counter()

        while self.running and self.source_running:
            if self.paused:
                time.sleep(0.05)
                continue

            with self.capture_lock:
                cap = self.capture
                if cap is None:
                    break
                success, frame = cap.read()

            if not success:
                if self.source_kind == "video":
                    # Reached the end of video: stop playback.
                    self.root.after(0, self.stop_source)
                else:
                    time.sleep(0.02)
                break

            # Apply selected orientation to video frames BEFORE storing
            # them for display and AI inference. Live camera stays unchanged.
            if self.source_kind == "video":
                rotation = self.rotation_var.get()
                if rotation == "90° Clockwise":
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                elif rotation == "90° Counterclockwise":
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif rotation == "180°":
                    frame = cv2.rotate(frame, cv2.ROTATE_180)

            with self.frame_lock:
                self.latest_frame = frame
                self.frame_count += 1

            fps_counter += 1
            elapsed = time.perf_counter() - fps_start
            if elapsed >= 1.0:
                self.camera_fps = fps_counter / elapsed
                fps_counter = 0
                fps_start = time.perf_counter()

            # Avoid reading a video much faster than its nominal frame rate.
            if self.source_kind == "video":
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps and fps > 0:
                    time.sleep(max(0.0, (1.0 / fps) - 0.001))

        print("[R-SAFE] Source reader stopped.")

    # ========================================================
    # INFERENCE THREAD
    # ========================================================

    def inference_loop(self):
        print("[R-SAFE] AI inference thread started.")

        while self.running:
            if not self.source_running or self.paused:
                time.sleep(0.05)
                continue

            with self.frame_lock:
                frame = self.latest_frame.copy() if self.latest_frame is not None else None
                current_count = self.frame_count

            if frame is None:
                time.sleep(0.05)
                continue

            if current_count - self.last_inference_frame < self.inference_interval:
                time.sleep(0.01)
                continue

            self.last_inference_frame = current_count
            start = time.perf_counter()

            try:
                detections = self.detector.detect(frame)
                elapsed = time.perf_counter() - start

                self.inference_time_ms = elapsed * 1000.0
                self.inference_fps = (1.0 / elapsed) if elapsed > 0 else 0.0

                helmet_count = 0
                no_helmet_count = 0

                for detection in detections:
                    name = detection.get("class_name", "").lower().replace("_", "-")
                    if name == "helmet":
                        helmet_count += 1
                    elif name in ("no-helmet", "no helmet", "nohelmet"):
                        no_helmet_count += 1

                with self.detection_lock:
                    self.latest_detections = detections
                    self.helmet_count = helmet_count
                    self.no_helmet_count = no_helmet_count

                print(
                    f"[R-SAFE] Inference: {self.inference_time_ms:.1f} ms | "
                    f"AI FPS: {self.inference_fps:.2f} | "
                    f"Helmet: {helmet_count} | No Helmet: {no_helmet_count}"
                )

            except Exception as error:
                print(f"[R-SAFE] Inference error: {error}")

            time.sleep(0.005)

        print("[R-SAFE] AI inference thread stopped.")

    # ========================================================
    # DRAW AND DISPLAY
    # ========================================================

    @staticmethod
    def draw_detections(frame, detections):
        output = frame.copy()
        frame_h, frame_w = output.shape[:2]

        for detection in detections:
            class_name = detection.get("class_name", "Unknown")
            confidence = detection.get("confidence", 0.0)
            bbox = detection.get("bbox")

            if bbox is None or len(bbox) != 4:
                continue

            x, y, width, height = map(int, bbox)
            x = max(0, min(x, frame_w - 1))
            y = max(0, min(y, frame_h - 1))
            x2 = max(0, min(x + width, frame_w - 1))
            y2 = max(0, min(y + height, frame_h - 1))

            normalized = class_name.lower().replace("_", "-")
            if normalized == "helmet":
                color = (0, 255, 0)
            elif normalized in ("no-helmet", "no helmet", "nohelmet"):
                color = (0, 0, 255)
            else:
                color = (255, 255, 0)

            cv2.rectangle(output, (x, y), (x2, y2), color, 2)
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(
                output, label, (x, max(y - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
                cv2.LINE_AA
            )

        return output

    def update_display(self):
        if not self.running:
            return

        with self.frame_lock:
            frame = self.latest_frame.copy() if self.latest_frame is not None else None

        if frame is not None:
            with self.detection_lock:
                detections = self.latest_detections.copy()
                helmet_count = self.helmet_count
                no_helmet_count = self.no_helmet_count

            frame = self.draw_detections(frame, detections)

            # Fit frame to the currently available display space while
            # preserving aspect ratio (no fixed 960x540 stretch).
            available_w = max(self.video_label.winfo_width(), 320)
            available_h = max(self.video_label.winfo_height(), 240)
            h, w = frame.shape[:2]
            scale = min(available_w / w, available_h / h)
            display_w = max(1, int(w * scale))
            display_h = max(1, int(h * scale))

            frame = cv2.resize(
                frame, (display_w, display_h),
                interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
            )
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            photo = ImageTk.PhotoImage(Image.fromarray(frame))

            self.video_label.configure(image=photo, text="")
            self.video_label.image = photo

            self.camera_fps_label.configure(text=f"Camera FPS: {self.camera_fps:.1f}")
            self.ai_fps_label.configure(text=f"AI FPS: {self.inference_fps:.2f}")
            self.inference_time_label.configure(
                text=f"Inference: {self.inference_time_ms:.1f} ms"
            )
            self.helmet_label.configure(text=f"Helmet: {helmet_count}")
            self.no_helmet_label.configure(text=f"No Helmet: {no_helmet_count}")

        self.root.after(DISPLAY_REFRESH_MS, self.update_display)

    # ========================================================
    # SETTINGS
    # ========================================================

    def change_resolution(self, event=None):
        try:
            new_size = int(self.resolution_var.get())
            self.inference_size = new_size
            self.detector.image_size = new_size
            print(f"[R-SAFE] AI resolution changed: {new_size} x {new_size}")
        except Exception as error:
            print(f"[R-SAFE] Resolution change error: {error}")

    def change_interval(self, event=None):
        try:
            self.inference_interval = int(self.interval_var.get())
            print(f"[R-SAFE] Inference interval: {self.inference_interval} frames")
        except Exception as error:
            print(f"[R-SAFE] Interval change error: {error}")

    def change_rotation(self, event=None):
        selected = self.rotation_var.get()
        print(f"[R-SAFE] Video rotation selected: {selected}")
        if self.source_kind == "video" and self.source_running:
            print("[R-SAFE] Rotation applies to newly read video frames.")

    # ========================================================
    # SHUTDOWN
    # ========================================================

    def close_application(self):
        print("[R-SAFE] Shutting down...")
        self.running = False
        self.stop_source()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = RSAFELiveDetectionGUI(root)
    root.mainloop()
