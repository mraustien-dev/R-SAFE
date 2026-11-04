import logging
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from rsafe.types import HelmetResult

logger = logging.getLogger("rsafe.detectors.helmet")


class HelmetDetector:
    """
    ONNX helmet detector for YOLO-style exports with two classes:
      0: Helmet
      1: No-Helmet

    Handles common output layouts:
      [1, 6, N] and [1, N, 6]
    """

    def __init__(
        self,
        model_path=None,
        confidence_threshold=0.20,
        iou_threshold=0.45,
        image_size=640,
        debug=True,
        providers=None,
    ):
        if model_path is None:
            model_path = (
                Path(__file__).resolve().parents[2]
                / "models"
                / "helmet"
                / "best.onnx"
            )

        self.model_path = Path(model_path)
        self.confidence_threshold = float(confidence_threshold)
        self.iou_threshold = float(iou_threshold)
        self.image_size = int(image_size)
        self.debug = debug
        self._printed_output_info = False

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=providers or ["CPUExecutionProvider"],
        )

        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        self.input_shape = model_input.shape

        self.class_names = {
            0: "Helmet",
            1: "No-Helmet",
        }

        if self.debug:
            logger.debug("[R-SAFE] ONNX model loaded")
            logger.debug(f"[R-SAFE] Input name: {self.input_name}")
            logger.debug(f"[R-SAFE] Input shape: {self.input_shape}")
            logger.debug(f"[R-SAFE] Confidence threshold: {self.confidence_threshold}")

        # Warn when ONNX expects a fixed size different from the selected size.
        if len(self.input_shape) == 4:
            model_h = self.input_shape[2]
            model_w = self.input_shape[3]
            if (
                isinstance(model_h, int)
                and isinstance(model_w, int)
                and (model_h != self.image_size or model_w != self.image_size)
            ):
                logger.warning(
                    "[R-SAFE WARNING] Selected image_size "
                    f"{self.image_size} does not match model input "
                    f"{model_w}x{model_h}. Using model's fixed input size."
                )
                self.image_size = int(model_h)

    def preprocess(self, image):
        original_height, original_width = image.shape[:2]

        scale = min(
            self.image_size / original_width,
            self.image_size / original_height,
        )

        new_width = int(round(original_width * scale))
        new_height = int(round(original_height * scale))

        resized = cv2.resize(image, (new_width, new_height))

        canvas = np.full(
            (self.image_size, self.image_size, 3),
            114,
            dtype=np.uint8,
        )

        pad_x = (self.image_size - new_width) // 2
        pad_y = (self.image_size - new_height) // 2

        canvas[
            pad_y:pad_y + new_height,
            pad_x:pad_x + new_width,
        ] = resized

        image_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        input_tensor = image_rgb.transpose(2, 0, 1)
        input_tensor = np.expand_dims(input_tensor, axis=0)
        input_tensor = np.ascontiguousarray(input_tensor, dtype=np.float32)
        input_tensor /= 255.0

        metadata = {
            "scale": scale,
            "pad_x": pad_x,
            "pad_y": pad_y,
            "original_width": original_width,
            "original_height": original_height,
        }

        return input_tensor, metadata

    def _normalize_output_layout(self, output):
        """
        Return prediction rows shaped [number_of_candidates, 6].
        Each row is [cx, cy, w, h, helmet_score, no_helmet_score].
        """
        output = np.asarray(output)

        # Remove batch dimension where present.
        if output.ndim == 3 and output.shape[0] == 1:
            output = output[0]

        if output.ndim != 2:
            raise ValueError(f"Unexpected ONNX output dimensions: {output.shape}")

        # Feature dimension should be 4 box values + 2 class scores = 6.
        if output.shape[0] == 6 and output.shape[1] != 6:
            output = output.T
        elif output.shape[1] == 6:
            pass
        elif output.shape[0] < output.shape[1] and output.shape[0] >= 6:
            output = output.T
        else:
            raise ValueError(
                "Could not determine ONNX output layout. "
                f"Expected a dimension of 6, got {output.shape}."
            )

        if output.shape[1] != 6:
            raise ValueError(
                f"Expected 6 values per prediction, got {output.shape}"
            )

        return output

    def detect(self, image):
        if image is None or image.size == 0:
            raise ValueError("Invalid input image.")

        input_tensor, metadata = self.preprocess(image)

        outputs = self.session.run(
            None,
            {self.input_name: input_tensor},
        )

        if not outputs:
            raise ValueError("ONNX model returned no outputs.")

        predictions = self._normalize_output_layout(outputs[0])

        class_scores = predictions[:, 4:6]
        best_class_ids = np.argmax(class_scores, axis=1)
        best_scores = np.max(class_scores, axis=1)

        if self.debug and not self._printed_output_info:
            logger.debug(f"[R-SAFE DEBUG] Raw output shape: {np.asarray(outputs[0]).shape}")
            logger.debug(f"[R-SAFE DEBUG] Normalized predictions: {predictions.shape}")
            logger.debug(
                "[R-SAFE DEBUG] Maximum Helmet score: "
                f"{float(np.max(class_scores[:, 0])):.4f}"
            )
            logger.debug(
                "[R-SAFE DEBUG] Maximum No-Helmet score: "
                f"{float(np.max(class_scores[:, 1])):.4f}"
            )
            logger.debug(
                "[R-SAFE DEBUG] Candidates at threshold "
                f"{self.confidence_threshold}: "
                f"{int(np.count_nonzero(best_scores >= self.confidence_threshold))}"
            )
            self._printed_output_info = True

        candidate_indices = np.where(
            best_scores >= self.confidence_threshold
        )[0]

        detections = []

        # Perform NMS independently for each class.
        for class_id in self.class_names:
            class_candidate_indices = [
                int(i)
                for i in candidate_indices
                if int(best_class_ids[i]) == class_id
            ]

            if not class_candidate_indices:
                continue

            class_boxes = []
            class_scores_for_nms = []
            mapped_boxes = []

            for prediction_index in class_candidate_indices:
                prediction = predictions[prediction_index]

                center_x, center_y, width, height = map(
                    float, prediction[:4]
                )

                x1 = center_x - width / 2
                y1 = center_y - height / 2
                x2 = center_x + width / 2
                y2 = center_y + height / 2

                x1 = (x1 - metadata["pad_x"]) / metadata["scale"]
                y1 = (y1 - metadata["pad_y"]) / metadata["scale"]
                x2 = (x2 - metadata["pad_x"]) / metadata["scale"]
                y2 = (y2 - metadata["pad_y"]) / metadata["scale"]

                x1 = int(np.clip(x1, 0, metadata["original_width"]))
                y1 = int(np.clip(y1, 0, metadata["original_height"]))
                x2 = int(np.clip(x2, 0, metadata["original_width"]))
                y2 = int(np.clip(y2, 0, metadata["original_height"]))

                if x2 <= x1 or y2 <= y1:
                    continue

                box_xywh = [x1, y1, x2 - x1, y2 - y1]
                mapped_boxes.append((prediction_index, box_xywh))
                class_boxes.append(box_xywh)
                class_scores_for_nms.append(float(best_scores[prediction_index]))

            if not class_boxes:
                continue

            kept_indices = cv2.dnn.NMSBoxes(
                class_boxes,
                class_scores_for_nms,
                self.confidence_threshold,
                self.iou_threshold,
            )

            if kept_indices is None or len(kept_indices) == 0:
                continue

            for kept_index in np.asarray(kept_indices).flatten():
                prediction_index, box = mapped_boxes[int(kept_index)]

                detections.append(
                    {
                        "class_id": class_id,
                        "class_name": self.class_names[class_id],
                        "confidence": float(best_scores[prediction_index]),
                        "bbox": box,
                    }
                )

        if self.debug:
            # Print count occasionally only when detections appear, keeping
            # the console useful without flooding it with zero-result lines.
            if detections:
                logger.debug(
                    "[R-SAFE DEBUG] Detections: "
                    + ", ".join(
                        f'{d["class_name"]}={d["confidence"]:.2f}'
                        for d in detections
                    )
                )

        return detections

    @staticmethod
    def draw_detections(image, detections):
        output = image.copy()

        for detection in detections:
            x, y, width, height = detection["bbox"]

            label = (
                f'{detection["class_name"]} '
                f'{detection["confidence"]:.2f}'
            )

            color = (
                (0, 200, 0)
                if detection["class_id"] == 0
                else (0, 0, 255)
            )

            cv2.rectangle(
                output,
                (x, y),
                (x + width, y + height),
                color,
                2,
            )

            cv2.putText(
                output,
                label,
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )

        return output


LABELS = {"Helmet": "helmet", "No-Helmet": "no_helmet"}


class HelmetClassifier:
    """Judge one head crop as helmet / no_helmet using the strongest helmet-model box."""

    name = "helmet-best.onnx"

    def __init__(self, model_path: Path, confidence: float, imgsz: int = 640, providers=None):
        self.detector = HelmetDetector(
            model_path=model_path, confidence_threshold=confidence, image_size=imgsz, debug=False,
            providers=providers,
        )

    def classify(self, crop: np.ndarray) -> HelmetResult | None:
        if crop is None or crop.size == 0:
            return None
        detections = self.detector.detect(crop)
        if not detections:
            return None
        best = max(detections, key=lambda d: d["confidence"])
        return HelmetResult(LABELS[best["class_name"]], best["confidence"])
