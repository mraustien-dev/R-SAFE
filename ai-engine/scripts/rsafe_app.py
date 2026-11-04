r"""R-SAFE window app: pick a video file, a laptop/USB camera, or a phone (IP Webcam),
rotate the picture if it is sideways, and watch the full engine work.

Start it by double-clicking ai-engine\R-SAFE.bat, or:
  .venv\Scripts\python scripts\rsafe_app.py [--config configs\demo.yaml]

Every flagged motorcycle is saved as an evidence folder (status "pending_review").
"""

import argparse
import logging
import os
import queue
import sys
import threading
import tkinter as tk
from datetime import datetime, timezone
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rsafe.app.session import AnalysisSession, VideoRecorder, find_cameras, normalize_stream_url  # noqa: E402
from rsafe.config import ConfigError, load_config  # noqa: E402
from rsafe.hardware import LEVELS  # noqa: E402
from rsafe.video.source import VideoSource, video_fps  # noqa: E402

AI_ENGINE_DIR = Path(__file__).resolve().parents[1]
VIDEO_TYPES = [("Videos", "*.mp4 *.avi *.mov *.mkv *.m4v *.wmv"), ("All files", "*.*")]
ROTATION_CHOICES = ["0°", "90° right", "180°", "90° left"]
ROTATION_DEGREES = {"0°": 0, "90° right": 90, "180°": 180, "90° left": 270}
POLL_MS = 40

BG, PANEL, TEXT, ACCENT, WARN = "#101820", "#1b2733", "#e8eef3", "#00d4ff", "#ff5a5a"


def _video_opens(path: str) -> bool:
    capture = cv2.VideoCapture(path)
    try:
        return capture.isOpened() and capture.read()[0]
    finally:
        capture.release()


class RsafeApp:
    def __init__(self, root: tk.Tk, cfg):
        self.root, self.cfg = root, cfg
        self.events: queue.Queue = queue.Queue()
        self.engine = None
        self.session: AnalysisSession | None = None
        self.worker: threading.Thread | None = None
        self.rotation = 0
        self.cameras: list[int] = []
        self._photo = None
        self._last_image = None
        self._closing = False
        self._source_name = ""
        self._auto_level = "low"
        self._frames_waiting = 0

        root.title("R-SAFE | Helmet Violation Detection")
        root.geometry("1280x760")
        root.minsize(900, 560)
        root.configure(bg=BG)
        root.protocol("WM_DELETE_WINDOW", self.close)
        self._build()
        self._set_status("Loading models (about 10 seconds)...")
        threading.Thread(target=self._load_engine, daemon=True).start()
        threading.Thread(target=self._scan_cameras, daemon=True).start()
        root.after(POLL_MS, self._poll)

    # ---------------------------------------------------------------- layout
    def _build(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TCombobox", fieldbackground=PANEL, background=PANEL, foreground=TEXT)

        bar = tk.Frame(self.root, bg=BG)
        bar.pack(fill="x", padx=10, pady=(10, 4))
        self.file_button = self._button(bar, "Choose video file", self.choose_file)
        self.file_button.pack(side="left")
        self._label(bar, "   Camera:").pack(side="left")
        self.camera_box = ttk.Combobox(bar, state="readonly", width=22, values=["Searching..."])
        self.camera_box.current(0)
        self.camera_box.pack(side="left", padx=4)
        self.camera_button = self._button(bar, "Connect camera", self.connect_camera)
        self.camera_button.pack(side="left")
        self._label(bar, "   Phone (IP Webcam):").pack(side="left")
        self.phone_entry = tk.Entry(bar, width=22, bg=PANEL, fg=TEXT, insertbackground=TEXT)
        self.phone_entry.insert(0, "192.168.1.5:8080")
        self.phone_entry.pack(side="left", padx=4)
        self.phone_button = self._button(bar, "Connect phone", self.connect_phone)
        self.phone_button.pack(side="left")

        bar2 = tk.Frame(self.root, bg=BG)
        bar2.pack(fill="x", padx=10, pady=4)
        self._label(bar2, "Rotate:").pack(side="left")
        self.rotation_box = ttk.Combobox(bar2, state="readonly", width=10, values=ROTATION_CHOICES)
        self.rotation_box.current(0)
        self.rotation_box.bind("<<ComboboxSelected>>", self._rotation_changed)
        self.rotation_box.pack(side="left", padx=4)
        self._label(bar2, "   Video filmed at (optional):").pack(side="left")
        self.filmed_entry = tk.Entry(bar2, width=18, bg=PANEL, fg=TEXT, insertbackground=TEXT)
        self.filmed_entry.pack(side="left", padx=4)
        self._label(bar2, "e.g. 2026-11-04 08:00", fg="#8899a6").pack(side="left")
        self.realtime_var = tk.BooleanVar(value=False)
        self.record_var = tk.BooleanVar(value=False)
        for text, variable in (("Real-time playback", self.realtime_var), ("Save analysed video", self.record_var)):
            tk.Checkbutton(bar2, text=text, variable=variable, bg=BG, fg=TEXT, selectcolor=PANEL,
                           activebackground=BG, activeforeground=TEXT, font=("Segoe UI", 9)).pack(side="left", padx=(12, 0))
        self.folder_button = self._button(bar2, "Open evidence folder", self.open_evidence_folder)
        self.level_box = ttk.Combobox(bar2, state="disabled", width=16, values=["Loading..."])
        self.level_box.current(0)
        self.level_box.bind("<<ComboboxSelected>>", self._level_changed)
        self.level_box.pack(side="right", padx=(0, 12))
        self._label(bar2, "Performance:").pack(side="right")
        self.folder_button.pack(side="right")
        self.stop_button = self._button(bar2, "Stop", self.stop, state="disabled")
        self.stop_button.pack(side="right", padx=4)
        self.pause_button = self._button(bar2, "Pause", self.toggle_pause, state="disabled")
        self.pause_button.pack(side="right")

        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=10, pady=4)
        side = tk.Frame(body, bg=PANEL, width=260)
        side.pack(side="right", fill="y", padx=(8, 0))
        side.pack_propagate(False)
        self._label(side, "Violations saved (double-click to open)", bg=PANEL).pack(anchor="w", padx=8, pady=(8, 4))
        self.saved_list = tk.Listbox(side, bg=BG, fg=WARN, selectbackground=ACCENT, borderwidth=0, highlightthickness=0)
        self.saved_list.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.saved_list.bind("<Double-Button-1>", self._open_selected)
        self.saved_folders: list[Path] = []
        self._label(side, "Blue = motorcycle\nGreen = helmet\nRed = no helmet\nGrey ? = not judged yet",
                    bg=PANEL, justify="left").pack(anchor="w", padx=8, pady=(0, 8))

        self.video = tk.Label(body, bg="black", fg="#8899a6", text="Choose a video file, a camera, or a phone",
                              font=("Segoe UI", 14))
        self.video.pack(side="left", fill="both", expand=True)
        self.video.bind("<Configure>", lambda event: self._show(self._last_image))

        status_bar = tk.Frame(self.root, bg=PANEL)
        status_bar.pack(fill="x", padx=10, pady=(4, 10))
        self.status = self._label(status_bar, "", bg=PANEL)
        self.status.pack(side="left", padx=8, pady=6)
        self.metrics = self._label(status_bar, "", bg=PANEL, fg=ACCENT)
        self.metrics.pack(side="right", padx=8, pady=6)

    def _button(self, parent, text, command, state="normal"):
        return tk.Button(parent, text=text, command=command, state=state, bg=ACCENT, fg="#06121a",
                         activebackground="#7fe9ff", relief="flat", padx=10, pady=3, font=("Segoe UI", 9, "bold"))

    def _label(self, parent, text, bg=BG, fg=TEXT, **options):
        return tk.Label(parent, text=text, bg=bg, fg=fg, font=("Segoe UI", 9), **options)

    # --------------------------------------------------------- background work
    def _load_engine(self) -> None:
        try:
            from rsafe.pipeline import Engine

            self.events.put(("ready", Engine(self.cfg)))
        except Exception as error:  # noqa: BLE001 - shown to the user
            logging.exception("[R-SAFE] Could not load the models")
            self.events.put(("fatal", f"Could not load the models: {error}"))

    def _scan_cameras(self) -> None:
        self.events.put(("cameras", find_cameras()))

    def _run_session(self, session: AnalysisSession, name: str) -> None:
        try:
            session.run()
            message = f"Finished {name}. Violations saved: {session.saved}."
            if session.recorder and session.recorder.path:
                message += f"  Analysed video: {session.recorder.path}"
            self.events.put(("done", message))
        except Exception as error:  # noqa: BLE001 - shown to the user
            logging.exception("[R-SAFE] Analysis stopped by an error")
            self.events.put(("done", f"Stopped by an error: {error}"))

    def _poll(self) -> None:
        latest_frame = None
        try:
            while True:
                kind, *payload = self.events.get_nowait()
                if kind == "frame":
                    self._frames_waiting -= 1
                    latest_frame = payload  # only draw the newest one
                elif kind == "saved":
                    self._add_saved(payload[0])
                elif kind == "ready":
                    self.engine = payload[0]
                    self._auto_level = self.engine.performance.level
                    self._fill_levels()
                    self._set_status("Ready. Choose a video file, a camera, or a phone.")
                elif kind == "cameras":
                    self._fill_cameras(payload[0])
                elif kind == "done":
                    self._session_ended(payload[0])
                elif kind == "fatal":
                    self._set_status(payload[0])
                    messagebox.showerror("R-SAFE", payload[0])
        except queue.Empty:
            pass
        if latest_frame:
            image, stats = latest_frame
            self._show(image)
            self.metrics.configure(text=f"Frames analysed: {stats['processed']}   "
                                        f"Speed: {stats['speed']:.1f} frames/s   Violations: {stats['saved']}")
            if self.session and not self.session.paused:
                self._set_status(f"Running: {self._source_name}")
        if self._closing and self.session is None:
            self.root.destroy()
            return
        self.root.after(POLL_MS, self._poll)

    # ------------------------------------------------------------------ sources
    def choose_file(self) -> None:
        path = filedialog.askopenfilename(title="Choose a traffic video", filetypes=VIDEO_TYPES)
        if not path:
            return
        filmed = self.filmed_entry.get().strip()
        try:
            start = datetime.fromisoformat(filmed) if filmed else datetime.now(timezone.utc)
        except ValueError:
            messagebox.showerror("R-SAFE", f"'Video filmed at' should look like 2026-11-04 08:00, not '{filmed}'.")
            return
        self._start(path, Path(path).name, start, video_fps(path))

    def connect_camera(self) -> None:
        if not self.cameras:
            messagebox.showinfo("R-SAFE", "No camera found on this laptop. Try Connect phone instead.")
            return
        index = self.cameras[self.camera_box.current()]
        self._start(str(index), f"camera {index}", datetime.now(timezone.utc), 0.0)

    def connect_phone(self) -> None:
        try:
            url = normalize_stream_url(self.phone_entry.get())
        except ValueError as error:
            messagebox.showerror("R-SAFE", str(error))
            return
        self._start(url, f"phone {url}", datetime.now(timezone.utc), 0.0)

    def _start(self, source_text: str, name: str, start: datetime, fps: float) -> None:
        if self.engine is None:
            messagebox.showinfo("R-SAFE", "The models are still loading. Try again in a few seconds.")
            return
        try:
            source = VideoSource(source_text, self.cfg.video.camera_id)
        except FileNotFoundError as error:
            messagebox.showerror("R-SAFE", str(error))
            return
        if source.info.kind == "file" and not _video_opens(source_text):
            messagebox.showerror("R-SAFE", "This video can't be opened. Try an .mp4 file "
                                           "(iPhone: set Camera > Formats to 'Most Compatible').")
            return
        is_file = source.info.kind == "file"
        realtime = is_file and self.realtime_var.get()
        recorder = None
        if self.record_var.get() and is_file and fps > 0:
            # analysed frames only (every stride-th) unless real-time, so it plays at normal speed
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            target = Path(self.cfg.evidence.output_dir).parent / "videos" / f"{Path(source_text).stem}_analysed_{stamp}.mp4"
            recorder = VideoRecorder(target, fps if realtime else fps / self.engine.cfg.video.frame_stride)
        self.session = AnalysisSession(
            self.engine.new_pipeline(), source,
            stride=self.engine.cfg.video.frame_stride, rotation=lambda: self.rotation,
            start_time=start, fps=fps, realtime=realtime, recorder=recorder,
            on_frame=lambda image, stats: self._put_frame(image, stats),
            on_saved=lambda folder: self.events.put(("saved", folder)),
        )
        self._source_name = name
        self._set_sources_enabled(False)
        self.pause_button.configure(state="normal", text="Pause")
        self.stop_button.configure(state="normal")
        if self.record_var.get() and not is_file:
            messagebox.showinfo("R-SAFE", "Save analysed video works for video files only; "
                                          "the camera will be analysed without saving a video.")
        self._set_status(f"Connecting to {name}..." if not is_file else f"Starting {name}...")
        self.worker = threading.Thread(target=self._run_session, args=(self.session, name), daemon=True)
        self.worker.start()

    def _put_frame(self, image, stats) -> None:
        # real-time modes send ~25 frames a second; drop extras the window hasn't drawn yet
        if self._frames_waiting < 3:
            self._frames_waiting += 1
            self.events.put(("frame", image, stats))

    # ---------------------------------------------------------------- controls
    def toggle_pause(self) -> None:
        if not self.session:
            return
        if self.session.paused:
            self.session.resume()
            self.pause_button.configure(text="Pause")
            self._set_status(f"Running: {self._source_name}")
        else:
            self.session.pause()
            self.pause_button.configure(text="Resume")
            self._set_status(f"Paused: {self._source_name}")

    def stop(self) -> None:
        if self.session:
            self.session.stop()
            self.pause_button.configure(state="disabled")
            self.stop_button.configure(state="disabled")
            self._set_status("Stopping and saving the last motorcycles...")

    def _session_ended(self, message: str) -> None:
        self.session = None
        self._set_sources_enabled(True)
        self.pause_button.configure(state="disabled", text="Pause")
        self.stop_button.configure(state="disabled")
        self._set_status(message)

    def _fill_levels(self) -> None:
        names = [f"Auto ({self._auto_level})"] + [level.capitalize() for level in LEVELS]
        self.level_box.configure(values=names, state="readonly")
        self.level_box.current(0)
        self._show_hardware()

    def _level_changed(self, event=None) -> None:
        choice = self.level_box.current()
        self.engine.set_level(self._auto_level if choice == 0 else list(LEVELS)[choice - 1])
        self._show_hardware()

    def _show_hardware(self) -> None:
        cfg = self.engine.cfg
        self.metrics.configure(text=f"{self.engine.performance.describe()}  "
                                    f"(every {cfg.video.frame_stride} frame(s), size {cfg.vehicle.imgsz})")

    def _rotation_changed(self, event=None) -> None:
        self.rotation = ROTATION_DEGREES[self.rotation_box.get()]

    def open_evidence_folder(self) -> None:
        folder = Path(self.cfg.evidence.output_dir)
        folder.mkdir(parents=True, exist_ok=True)
        os.startfile(folder)

    def _open_selected(self, event=None) -> None:
        selection = self.saved_list.curselection()
        if selection:
            os.startfile(self.saved_folders[selection[0]])

    def close(self) -> None:
        """Closing the window first saves the motorcycles still being watched."""
        self._closing = True
        if self.session:
            self.stop()
            self._set_status("Saving the last motorcycles, then closing...")

    # ----------------------------------------------------------------- helpers
    def _fill_cameras(self, cameras: list[int]) -> None:
        self.cameras = cameras
        names = [f"Camera {i}" + (" (built-in)" if i == 0 else "") for i in cameras] or ["No camera found"]
        self.camera_box.configure(values=names)
        self.camera_box.current(0)

    def _add_saved(self, folder) -> None:
        folder = Path(folder)
        self.saved_folders.append(folder)
        self.saved_list.insert("end", folder.name)
        self.saved_list.see("end")

    def _set_sources_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        for button in (self.file_button, self.camera_button, self.phone_button):
            button.configure(state=state)
        if self.engine is not None:  # the level applies from the next video/camera on
            self.level_box.configure(state="readonly" if enabled else "disabled")

    def _set_status(self, text: str) -> None:
        self.status.configure(text=text)

    def _show(self, image) -> None:
        """Fit the frame into the video area, keeping its shape."""
        self._last_image = image
        if image is None:
            return
        width, height = self.video.winfo_width(), self.video.winfo_height()
        if width < 10 or height < 10:
            return
        scale = min(width / image.shape[1], height / image.shape[0])
        size = (max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale)))
        rgb = cv2.cvtColor(cv2.resize(image, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.video.configure(image=self._photo, text="")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=AI_ENGINE_DIR / "configs" / "default.yaml")
    args = parser.parse_args()
    try:
        cfg = load_config(args.config)
    except (ConfigError, FileNotFoundError) as error:
        print(f"[R-SAFE] {error}", file=sys.stderr)
        return 2
    logging.basicConfig(level=cfg.logging.level, format="[R-SAFE] %(levelname)s %(name)s: %(message)s")
    root = tk.Tk()
    RsafeApp(root, cfg)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
