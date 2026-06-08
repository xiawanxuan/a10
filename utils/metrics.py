import time
import numpy as np
from collections import defaultdict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)


class PerformanceMetrics:
    def __init__(self):
        self.inference_times = []
        self.batch_times = []
        self.memory_usage = []
    
    def start_timer(self):
        self._start_time = time.time()
    
    def end_timer(self):
        elapsed = time.time() - self._start_time
        self.inference_times.append(elapsed)
        return elapsed
    
    def add_batch_time(self, batch_time):
        self.batch_times.append(batch_time)
    
    def get_inference_stats(self):
        if not self.inference_times:
            return {}
        
        times = np.array(self.inference_times)
        return {
            "count": len(times),
            "total_time": float(times.sum()),
            "avg_time": float(times.mean()),
            "min_time": float(times.min()),
            "max_time": float(times.max()),
            "std_time": float(times.std()),
            "fps": float(1.0 / times.mean()) if times.mean() > 0 else 0,
        }
    
    def reset(self):
        self.inference_times = []
        self.batch_times = []
        self.memory_usage = []


class ClassificationMetrics:
    def __init__(self, class_names=None):
        self.class_names = class_names
        self.y_true = []
        self.y_pred = []
        self.y_scores = []
    
    def update(self, y_true, y_pred, y_scores=None):
        self.y_true.extend(y_true)
        self.y_pred.extend(y_pred)
        if y_scores is not None:
            if isinstance(y_scores, np.ndarray):
                self.y_scores.extend(y_scores.tolist())
            else:
                self.y_scores.extend(y_scores)
    
    def compute_metrics(self):
        if not self.y_true:
            return {}
        
        y_true = np.array(self.y_true)
        y_pred = np.array(self.y_pred)
        
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
            "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
            "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        }
        
        if self.class_names:
            labels = list(range(len(self.class_names)))
            per_class = classification_report(
                y_true, y_pred,
                labels=labels,
                target_names=self.class_names,
                output_dict=True,
                zero_division=0,
            )
            metrics["per_class"] = per_class
        
        if self.class_names:
            labels = list(range(len(self.class_names)))
            metrics["confusion_matrix"] = confusion_matrix(
                y_true, y_pred, labels=labels
            ).tolist()
        else:
            metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
        
        return metrics
    
    def reset(self):
        self.y_true = []
        self.y_pred = []
        self.y_scores = []


class DetectionMetrics:
    def __init__(self, iou_threshold=0.5, class_names=None):
        self.iou_threshold = iou_threshold
        self.class_names = class_names
        self.predictions = []
        self.ground_truths = []
    
    def add_detections(self, image_id, predictions, ground_truths=None):
        for pred in predictions:
            pred["image_id"] = image_id
            self.predictions.append(pred)
        
        if ground_truths:
            for gt in ground_truths:
                gt["image_id"] = image_id
                self.ground_truths.append(gt)
    
    @staticmethod
    def compute_iou(box1, box2):
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        inter_x1 = max(x1_1, x1_2)
        inter_y1 = max(y1_1, y1_2)
        inter_x2 = min(x2_1, x2_2)
        inter_y2 = min(y2_1, y2_2)
        
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        union_area = box1_area + box2_area - inter_area
        
        iou = inter_area / union_area if union_area > 0 else 0
        return iou
    
    def compute_metrics(self):
        if not self.predictions:
            return {
                "num_predictions": 0,
                "num_ground_truths": 0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "map": 0.0,
            }
        
        class_metrics = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
        
        gt_by_image = defaultdict(list)
        for gt in self.ground_truths:
            gt_by_image[gt["image_id"]].append(gt)
        
        pred_by_image = defaultdict(list)
        for pred in self.predictions:
            pred_by_image[pred["image_id"]].append(pred)
        
        for image_id in set(list(gt_by_image.keys()) + list(pred_by_image.keys())):
            img_gts = gt_by_image.get(image_id, [])
            img_preds = pred_by_image.get(image_id, [])
            
            img_preds_sorted = sorted(img_preds, key=lambda x: x["confidence"], reverse=True)
            
            matched_gt = set()
            
            for pred in img_preds_sorted:
                pred_class = pred["class_id"]
                best_iou = 0
                best_gt_idx = -1
                
                for gt_idx, gt in enumerate(img_gts):
                    if gt_idx in matched_gt:
                        continue
                    if gt["class_id"] != pred_class:
                        continue
                    
                    iou = self.compute_iou(pred["bbox"], gt["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                if best_iou >= self.iou_threshold and best_gt_idx >= 0:
                    class_metrics[pred_class]["tp"] += 1
                    matched_gt.add(best_gt_idx)
                else:
                    class_metrics[pred_class]["fp"] += 1
            
            for gt_idx, gt in enumerate(img_gts):
                if gt_idx not in matched_gt:
                    class_metrics[gt["class_id"]]["fn"] += 1
        
        total_tp = sum(m["tp"] for m in class_metrics.values())
        total_fp = sum(m["fp"] for m in class_metrics.values())
        total_fn = sum(m["fn"] for m in class_metrics.values())
        
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        per_class_ap = {}
        for class_id, metrics_cls in class_metrics.items():
            tp = metrics_cls["tp"]
            fp = metrics_cls["fp"]
            fn = metrics_cls["fn"]
            class_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            class_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            per_class_ap[class_id] = class_precision * class_recall
        
        map_score = np.mean(list(per_class_ap.values())) if per_class_ap else 0.0
        
        return {
            "iou_threshold": self.iou_threshold,
            "num_predictions": len(self.predictions),
            "num_ground_truths": len(self.ground_truths),
            "num_classes": len(class_metrics),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "map": float(map_score),
            "per_class": {
                (self.class_names[cid] if self.class_names and cid < len(self.class_names) else str(cid)): {
                    "tp": m["tp"],
                    "fp": m["fp"],
                    "fn": m["fn"],
                }
                for cid, m in class_metrics.items()
            },
        }
    
    def reset(self):
        self.predictions = []
        self.ground_truths = []
