import cv2
import json
from ultralytics import YOLO
from typing import List, Dict, Any


class VisionAnalyzer:
    def __init__(self, model_path: str = "yolov8n.pt"):
        self.model = YOLO(model_path)

    def analyze(self, screenshot_path: str) -> List[Dict[str, Any]]:
        """
        Возвращает список областей интерфейса: {label, bbox, confidence}
        bbox = [x1, y1, x2, y2]
        """
        results = self.model(screenshot_path)
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(
                    {
                        "label": r.names[int(box.cls)],
                        "bbox": [x1, y1, x2, y2],
                        "confidence": float(box.conf),
                    }
                )
        return detections

    def match_to_dom(
        self, detections: List[Dict[str, Any]], dom_nodes: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Сопоставляет области с DOM-элементами по пересечению bbox.
        dom_nodes = [{"id": "...", "bbox": [x1,y1,x2,y2], "tag": "button"}]
        """
        matched = []
        for det in detections:
            dx1, dy1, dx2, dy2 = det["bbox"]
            dbest, best_iou = None, 0
            for node in dom_nodes:
                nx1, ny1, nx2, ny2 = node.get("bbox", [0, 0, 0, 0])
                iou = self._iou([dx1, dy1, dx2, dy2], [nx1, ny1, nx2, ny2])
                if iou > best_iou:
                    dbest, best_iou = node, iou
            if dbest and best_iou > 0.3:  # порог пересечения
                det["dom_match"] = dbest
            matched.append(det)
        return matched

    @staticmethod
    def _iou(boxA, boxB) -> float:
        """Intersection over Union"""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return inter / float(boxAArea + boxBArea - inter + 1e-6)

    def filter_by_labels(self, detections, allowed: List[str]):
        """Фильтруем только нужные типы объектов"""
        return [d for d in detections if d["label"].lower() in allowed]

    def normalize_bboxes(self, detections, width: int, height: int):
        """Нормализуем bbox (0..1)"""
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            d["bbox_norm"] = [x1 / width, y1 / height, x2 / width, y2 / height]
        return detections

    def hierarchy(self, detections):
        """Группируем элементы по вложенности"""
        result = []
        for d in detections:
            parents = []
            dx1, dy1, dx2, dy2 = d["bbox"]
            for other in detections:
                if other is d:
                    continue
                ox1, oy1, ox2, oy2 = other["bbox"]
                if ox1 <= dx1 and oy1 <= dy1 and ox2 >= dx2 and oy2 >= dy2:
                    parents.append(other["label"])
            d["parents"] = parents
            result.append(d)
        return result
