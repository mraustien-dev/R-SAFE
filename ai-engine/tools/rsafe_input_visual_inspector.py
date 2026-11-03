import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from pathlib import Path

def rotate_frame(frame, choice):
    if choice == "90 CW":
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if choice == "90 CCW":
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if choice == "180":
        return cv2.rotate(frame, cv2.ROTATE_180)
    return frame

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    h, w = img.shape[:2]
    nh, nw = new_shape
    scale = min(nh / h, nw / w)
    resized_w, resized_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    pad_w, pad_h = nw - resized_w, nh - resized_h
    left, top = int(round(pad_w / 2 - 0.1)), int(round(pad_h / 2 - 0.1))
    right, bottom = nw - resized_w - left, nh - resized_h - top
    out = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return out

def main():
    root = tk.Tk()
    root.withdraw()
    video = filedialog.askopenfilename(
        title="Choose the traffic video",
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
    )
    if not video:
        return

    rotation = simpledialog.askstring(
        "Video rotation",
        "Enter rotation exactly: 0, 90 CW, 90 CCW, or 180",
        initialvalue="0"
    )
    if rotation is None:
        return
    rotation = rotation.strip().upper()
    if rotation not in ("0", "90 CW", "90 CCW", "180"):
        messagebox.showerror("Invalid rotation", "Use 0, 90 CW, 90 CCW, or 180.")
        return

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open video.")
        return

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if total <= 0:
        messagebox.showerror("Error", "Could not read video frame count.")
        cap.release()
        return

    out_dir = Path(video).parent / "rsafe_input_inspection"
    out_dir.mkdir(exist_ok=True)
    samples = [0.10, 0.25, 0.50, 0.75, 0.90]
    saved = []

    for fraction in samples:
        frame_index = min(total - 1, int(total * fraction))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            continue
        frame = rotate_frame(frame, rotation)
        original_path = out_dir / f"frame_{int(fraction*100):02d}_rotated.jpg"
        cv2.imwrite(str(original_path), frame)

        model_input = letterbox(frame, (640, 640))
        input_path = out_dir / f"frame_{int(fraction*100):02d}_640_letterbox.jpg"
        cv2.imwrite(str(input_path), model_input)
        saved.extend([str(original_path), str(input_path)])

    cap.release()
    messagebox.showinfo(
        "Inspection images saved",
        f"Saved frame and 640x640 letterbox previews in:\n{out_dir}\n\n"
        "Open the files ending in _640_letterbox.jpg and check whether riders/helmets are clear."
    )

if __name__ == "__main__":
    main()
