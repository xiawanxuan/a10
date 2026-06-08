import os
import torch
import numpy as np
from PIL import Image
import cv2


class YOLOv5Detector:
    def __init__(
        self,
        model_path=None,
        conf_threshold=0.25,
        iou_threshold=0.45,
        image_size=640,
        device="cpu",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.image_size = image_size
        self.model = None
        self.class_names = []
        
        self._load_model(model_path)
    
    def _load_model(self, model_path=None):
        if model_path and os.path.exists(model_path):
            self.model = torch.hub.load(
                "ultralytics/yolov5",
                "custom",
                path=model_path,
                device=str(self.device),
            )
        else:
            self.model = torch.hub.load(
                "ultralytics/yolov5",
                "yolov5n",
                pretrained=True,
                device=str(self.device),
            )
        
        self.model.conf = self.conf_threshold
        self.model.iou = self.iou_threshold
        self.class_names = self.model.names
        print(f"YOLOv5 model loaded, classes: {len(self.class_names)}")
    
    def detect(self, image):
        if isinstance(image, str):
            img = image
        elif isinstance(image, np.ndarray):
            img = image
        elif isinstance(image, Image.Image):
            img = np.array(image)
        else:
            raise ValueError("Unsupported image type")
        
        results = self.model(img, size=self.image_size)
        
        detections = []
        for det in results.xyxy[0]:
            x1, y1, x2, y2 = det[:4].cpu().numpy().tolist()
            conf = float(det[4].cpu().item())
            cls_id = int(det[5].cpu().item())
            cls_name = self.class_names[cls_id] if cls_id < len(self.class_names) else str(cls_id)
            
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
                "class_id": cls_id,
                "class_name": cls_name,
            })
        
        return detections
    
    def detect_batch(self, images):
        results_list = []
        for image in images:
            results = self.detect(image)
            results_list.append(results)
        return results_list
    
    def set_conf_threshold(self, threshold):
        self.conf_threshold = threshold
        self.model.conf = threshold
    
    def set_iou_threshold(self, threshold):
        self.iou_threshold = threshold
        self.model.iou = threshold
