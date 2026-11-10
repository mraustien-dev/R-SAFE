r"""Export the R-SAFE models for the phone app as fast 8-bit (int8 QDQ) ONNX files.

Measured on the test phone (Redmi Note 14 Pro+, CPU): 8-bit is ~3x faster than 32-bit with
the same detections. Recipe (each step fixed a measured accuracy problem):
  * fixed input sizes: vehicle 480, helmet 320, plate 640 (team speed study)
  * 8-bit weights (per channel) and activations, static QDQ
  * calibration ONLY on images the full-precision model is confident about; otherwise the
    class-score range never sees a positive value and every score is capped at 0.50
  * the detection head's decode steps stay float; the 3 class-score (logit) outputs get
    16 bits with a fixed range [-48, 16]
  * Ultralytics metadata (class names) is kept, so the files also run in the laptop engine

Usage (from ai-engine\, with the ai-engine venv):
  .venv\Scripts\python -I tools\export_mobile_models.py
  .venv\Scripts\python -I tools\export_mobile_models.py --out <folder> --models helmet

Writes vehicle.onnx, helmet.onnx, plate.onnx and models.json (hashes, sizes, recipe) to --out
(default: the phone app's model folder). Deterministic: same inputs give the same files.
"""

import argparse
import csv
import gc
import hashlib
import json
import random
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image
from onnxruntime.quantization import CalibrationDataReader, QuantType, quantize
from onnxruntime.quantization.execution_providers.qnn import get_qnn_qdq_config
from onnxruntime.tools.onnx_model_utils import fix_output_shapes, make_input_shape_fixed

AI_ENGINE = Path(__file__).resolve().parents[1]
ROOT = AI_ENGINE.parent
DATA = ROOT / "assets" / "datasets"
DEFAULT_OUT = ROOT / "mobile/modules/rsafe-detector/android/src/main/assets/models"
TOOL_VERSION = "1"

# name: (source model, input size, classes, calibration images)
# Calibration counts are not multiples of 3 (CalibMaxIntermediateOutputs=3: ORT raises
# "No data is collected" when the last flush leaves nothing).
MODELS = {
    # ONNX export of yolo11s.pt (made once with:
    #   yolo export model=models/vehicle/yolo11s.pt format=onnx dynamic=True simplify=True opset=17)
    # A .pt path also works here but loads PyTorch (~1.5 GB of memory).
    "vehicle": (AI_ENGINE / "models/vehicle/export/yolo11s.onnx", 480, 80, 100),
    "helmet": (AI_ENGINE / "models/helmet/best.onnx", 320, 2, 160),
    "plate": (AI_ENGINE / "models/plate/plate.onnx", 640, 1, 64),
}
HOLDOUT = 50  # last N of each image pool: never used for calibration (used by --check)
LOGIT_RANGE = (-48.0, 16.0)


def letterbox(img: Image.Image, s: int) -> np.ndarray:
    """Same letterbox as the phone (Yolo.kt): fit, centre, grey 114, CHW floats 0..1."""
    img = img.convert("RGB")
    r = min(s / img.width, s / img.height)
    w, h = max(1, round(img.width * r)), max(1, round(img.height * r))
    canvas = Image.new("RGB", (s, s), (114, 114, 114))
    canvas.paste(img.resize((w, h), Image.BILINEAR), ((s - w) // 2, (s - h) // 2))
    return (np.asarray(canvas, np.float32) / 255.0).transpose(2, 0, 1)[None]


# ---- image pools from the R-SAFE datasets -------------------------------------------------
def yangon_boxes():
    """(frame path, motorcycle+rider box) from the HELMET (Yangon) annotations."""
    out = []
    ann = DATA / "HELMET/annotation/annotation"
    for clip in sorted((DATA / "HELMET/part_7/part_7").iterdir()):
        f = ann / f"{clip.name}.csv"
        if not f.exists():
            continue
        with open(f, newline="") as fh:
            for row in csv.DictReader(fh):
                img = clip / f"{int(row['frame_id']):02d}.jpg"
                if img.exists():
                    x, y, w, h = (float(row[k]) for k in "xywh")
                    out.append((img, (x, y, x + w, y + h)))
    return out


def head_crop(img, box):
    """Top of the rider box, padded: the same idea as headCropRect() in the app."""
    x1, y1, x2, y2 = box
    h = (y2 - y1) * 0.40
    pad = (x2 - x1) * 0.20
    return img.crop((max(0, x1 - pad), max(0, y1 - pad), min(img.width, x2 + pad), min(img.height, y1 + h)))


def plate_rows():
    with open(DATA / "RideSafe-400/plate_eval/plates.csv", newline="") as fh:
        return list(csv.DictReader(fh))


def plate_region(row):
    img = Image.open(DATA / "RideSafe-400/plate_eval/images" / row["image"])
    return img.crop(tuple(float(row[k]) for k in ("mx1", "my1", "mx2", "my2")))


def pools(rng: random.Random):
    boxes = yangon_boxes()
    rng.shuffle(boxes)
    frames = sorted({p for p, _ in boxes})
    rng.shuffle(frames)
    plates = plate_rows()
    rng.shuffle(plates)
    return {
        "vehicle": [lambda p=p: Image.open(p) for p in frames],
        "helmet": [lambda b=b: head_crop(Image.open(b[0]), b[1]) for b in boxes],
        "plate": [lambda r=r: plate_region(r) for r in plates],
    }


class Reader(CalibrationDataReader):
    def __init__(self, loaders, s):
        self.it = iter(loaders)
        self.s = s

    def get_next(self):
        nxt = next(self.it, None)
        return None if nxt is None else {"images": letterbox(nxt(), self.s)}


# ---- conversion ---------------------------------------------------------------------------
def source_onnx(name: str, src: Path, size: int, tmp: Path) -> Path:
    """The model as ONNX (the vehicle model is exported from the Ultralytics .pt)."""
    if src.suffix == ".onnx":
        return src
    from ultralytics import YOLO  # only needed for the .pt vehicle model

    out = Path(YOLO(str(src)).export(format="onnx", imgsz=size, dynamic=False, simplify=True, opset=17))
    dst = tmp / f"{name}_src.onnx"
    shutil.move(str(out), dst)
    return dst


def confident(name, model_path, size, nc, pool, need):
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    chosen, checked = [], 0
    for load in pool[:-HOLDOUT]:
        checked += 1
        if sess.run(None, {"images": letterbox(load(), size)})[0][0][4:4 + nc].max() >= 0.5:
            chosen.append(load)
            if len(chosen) == need:
                break
    print(f"  calibration: {len(chosen)} confident images (checked {checked})")
    if len(chosen) % 3 == 0:  # see MODELS comment
        chosen = chosen[:-1]
    return chosen


def export(name, src, size, nc, need, pool, out: Path, tmp: Path) -> dict:
    print(f"{name}: {src.name} -> {size}x{size} int8")
    m = onnx.load(str(source_onnx(name, src, size, tmp)))
    metadata = {p.key: p.value for p in m.metadata_props}
    make_input_shape_fixed(m.graph, "images", [1, 3, size, size])
    fix_output_shapes(m)
    del m.metadata_props[:]
    fp32 = tmp / f"{name}_fp32.onnx"
    onnx.save(m, str(fp32))

    g = m.graph
    head = "/" + g.node[-1].name.split("/")[1] + "/"
    exclude = [n.name for n in g.node
               if n.name.startswith(head) and "/cv2." not in n.name and "/cv3." not in n.name]
    logits = [n.output[0] for n in g.node
              if n.op_type == "Conv" and n.name.startswith(head + "cv3.") and n.name.endswith(".2/Conv")]
    assert len(logits) == 3, f"expected 3 class-score convs in {head}, found {len(logits)}"
    del m, g
    gc.collect()

    calib = confident(name, fp32, size, nc, pool, need)
    overrides = {t: [{"quant_type": QuantType.QUInt16,
                      "rmin": np.float32(LOGIT_RANGE[0]), "rmax": np.float32(LOGIT_RANGE[1])}] for t in logits}
    cfg = get_qnn_qdq_config(str(fp32), Reader(calib, size), activation_type=QuantType.QUInt8,
                             weight_type=QuantType.QUInt8, per_channel=True,
                             nodes_to_exclude=exclude, init_overrides=overrides)
    cfg.extra_options["CalibMaxIntermediateOutputs"] = 3  # bounded memory
    q = tmp / f"{name}_int8.onnx"
    quantize(str(fp32), str(q), cfg)

    # restore the Ultralytics metadata, with the real (fixed) input size
    qm = onnx.load(str(q))
    metadata["imgsz"] = str([size, size])
    metadata["dynamic"] = "False"
    for k, v in metadata.items():
        p = qm.metadata_props.add()
        p.key, p.value = k, v
    dst = out / f"{name}.onnx"
    onnx.save(qm, str(dst))
    del qm
    gc.collect()
    data = dst.read_bytes()
    return {
        "file": dst.name,
        "source": str(src.relative_to(ROOT)).replace("\\", "/"),
        "source_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "input_size": size,
        "classes": nc,
        "precision": "int8 (QDQ, per-channel weights, 16-bit class scores, float decode)",
        "calibration_images": len(calib),
        "_fp32": fp32,  # for --check, removed before writing the manifest
    }


def check(name, info, pool):
    """Do the int8 files keep the full-precision detections on held-out images?"""
    size, nc = info["input_size"], info["classes"]
    a = ort.InferenceSession(str(info["_fp32"]), providers=["CPUExecutionProvider"])
    b = ort.InferenceSession(str(info["_dst"]), providers=["CPUExecutionProvider"])
    conf = 0.35 if name == "vehicle" else 0.25
    both = either = 0
    top_a, top_b = [], []
    for load in pool[-HOLDOUT:][:20]:
        x = letterbox(load(), size)
        sa = a.run(None, {"images": x})[0][0][4:4 + nc].max(0)
        sb = b.run(None, {"images": x})[0][0][4:4 + nc].max(0)
        both += int(((sa >= conf) & (sb >= conf)).sum())
        either += int(((sa >= conf) | (sb >= conf)).sum())
        top_a.append(sa.max())
        top_b.append(sb.max())
    agree = both / either if either else 1.0
    print(f"  check on 20 held-out images: detections agree {agree:.0%}, "
          f"mean top score fp32 {np.mean(top_a):.2f} vs int8 {np.mean(top_b):.2f}")
    return round(agree, 3)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rng = random.Random(0)
    image_pools = pools(rng)
    manifest_path = args.out / "models.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"models": {}}

    with tempfile.TemporaryDirectory(prefix="rsafe_export_") as t:
        tmp = Path(t)
        for name in args.models:
            src, size, nc, need = MODELS[name]
            info = export(name, src, size, nc, need, image_pools[name], args.out, tmp)
            info["_dst"] = args.out / info["file"]
            info["check_agreement"] = check(name, info, image_pools[name])
            manifest["models"][name] = {k: v for k, v in info.items() if not k.startswith("_")}
            gc.collect()

    manifest.update({
        "tool": "ai-engine/tools/export_mobile_models.py",
        "tool_version": TOOL_VERSION,
        "onnxruntime": ort.__version__,
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    manifest["models"]["ocr"] = {"file": "mlkit-text-latin", "precision": "ML Kit bundled model"}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print("wrote", manifest_path)


if __name__ == "__main__":
    main()
