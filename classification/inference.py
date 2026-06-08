import torch
import numpy as np
from PIL import Image
import cv2

from .model import build_resnet18
from data.transforms import preprocess_image


class ClassifierInference:
    def __init__(self, model_path, num_classes=10, class_names=None, device="cpu"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.class_names = class_names or [str(i) for i in range(num_classes)]
        
        self.model = build_resnet18(num_classes=num_classes, pretrained=False)
        self._load_model(model_path)
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def _load_model(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
    
    def predict(self, image, top_k=5):
        top_k = min(top_k, self.num_classes)
        top_k = max(top_k, 1)
        
        tensor = preprocess_image(image, image_size=224)
        tensor = tensor.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = torch.softmax(outputs, dim=1)
            top_probs, top_indices = torch.topk(probabilities, top_k, dim=1)
        
        results = []
        for i in range(top_k):
            class_idx = top_indices[0][i].item()
            prob = top_probs[0][i].item()
            results.append({
                "class_id": class_idx,
                "class_name": self.class_names[class_idx] if class_idx < len(self.class_names) else str(class_idx),
                "confidence": prob,
            })
        
        return results
    
    def predict_batch(self, images, top_k=5):
        results_list = []
        for image in images:
            results = self.predict(image, top_k=top_k)
            results_list.append(results)
        return results_list
