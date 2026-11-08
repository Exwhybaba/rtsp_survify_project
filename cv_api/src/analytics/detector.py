"""Object detector abstraction.

``Detector`` is a tiny interface so the rest of the pipeline never cares
which model produced a box. Today there is one real implementation,
``YoloV8Detector``, backed by Ultralytics YOLOv8. Swapping in a different
model later (a custom-trained checkpoint, a different architecture, a
cloud vision API) means writing a new class with the same ``detect``
method -- nothing downstream changes.

The Ultralytics import is deliberately deferred to ``__init__`` (not the
module top level) so this module can be imported -- and ``Detection``
constructed, and a fake detector unit-tested -- in environments/tests
that do not have ``ultralytics``/``torch`` installed and never download
model weights.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from src.analytics.tracker import Detection

# COCO classes relevant to security/site-monitoring use cases.
DEFAULT_CLASSES_OF_INTEREST = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
}


class Detector(ABC):
    """Interface every detector implementation must satisfy."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run inference on a single BGR frame and return detections."""
        raise NotImplementedError


class YoloV8Detector(Detector):
    """Ultralytics YOLOv8 detector, filtered to a configurable class set.

    Parameters
    ----------
    model_path:
        Path or model name understood by ``ultralytics.YOLO`` (e.g.
        ``"yolov8n.pt"``, which auto-downloads the pretrained nano
        checkpoint on first use).
    confidence_threshold:
        Minimum detection confidence to keep.
    classes_of_interest:
        Set of COCO class names to keep; everything else is dropped. Pass
        ``None`` to keep every class the model produces.
    device:
        Ultralytics device string, e.g. ``"cpu"``, ``"cuda:0"``.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.4,
        classes_of_interest: set[str] | None = None,
        device: str = "cpu",
        image_size: int = 640,
    ) -> None:
        try:
            from ultralytics import YOLO  # noqa: PLC0415  # pylint: disable=import-outside-toplevel
        except ImportError as exc:  # pragma: no cover - exercised only without the dep
            raise ImportError(
                "The 'ultralytics' package is required for YoloV8Detector. "
                "Install it with `pip install ultralytics` (see requirements.txt)."
            ) from exc

        self._model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.classes_of_interest = (
            DEFAULT_CLASSES_OF_INTEREST if classes_of_interest is None else classes_of_interest
        )
        self.device = device
        self.image_size = image_size

    def detect(self, frame: np.ndarray) -> list[Detection]:
        results = self._model.predict(
            frame,
            conf=self.confidence_threshold,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )
        return list(self._to_detections(results))

    def _to_detections(self, results: list[Any]) -> list[Detection]:
        detections: list[Detection] = []
        if not results:
            return detections

        result = results[0]
        names = result.names  # {class_id: class_name}
        boxes = result.boxes
        if boxes is None:
            return detections

        for box in boxes:
            class_id = int(box.cls[0])
            class_name = names.get(class_id, str(class_id))
            if self.classes_of_interest is not None and class_name not in self.classes_of_interest:
                continue
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            detections.append(
                Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=confidence,
                    class_name=class_name,
                )
            )
        return detections
