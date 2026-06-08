import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import Image
import cv2

from config.settings import Config
from utils import (
    draw_detections,
    draw_classification,
    PerformanceMetrics,
    save_image,
)


def test_classification():
    print("=" * 50)
    print("Testing Classification...")
    print("=" * 50)
    
    from classification import ClassifierInference
    
    model_path = Config.CLASSIFICATION_MODEL_PATH
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please train the model first: python main.py train --use-cifar10")
        return False
    
    classifier = ClassifierInference(
        model_path=model_path,
        num_classes=len(Config.CLASS_NAMES),
        class_names=Config.CLASS_NAMES,
        device=Config.DEVICE,
    )
    
    test_image = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
    test_image = Image.fromarray(test_image)
    
    perf = PerformanceMetrics()
    
    print("\nRunning classification inference...")
    perf.start_timer()
    results = classifier.predict(test_image, top_k=5)
    inference_time = perf.end_timer()
    
    print(f"Inference time: {inference_time:.4f}s")
    print("\nTop 5 predictions:")
    for i, result in enumerate(results):
        print(f"  {i + 1}. {result['class_name']}: {result['confidence']:.4f}")
    
    visual_img = draw_classification(
        np.array(test_image), results, top_k=5
    )
    
    output_path = os.path.join(Config.OUTPUTS_DIR, "classification_result.jpg")
    save_image(visual_img, output_path)
    print(f"\nVisualization saved to: {output_path}")
    
    return True


def test_detection():
    print("\n" + "=" * 50)
    print("Testing Detection...")
    print("=" * 50)
    
    try:
        from detection import YOLOv5Detector
    except Exception as e:
        print(f"Failed to import YOLOv5Detector: {e}")
        return False
    
    model_path = Config.DETECTION_MODEL_PATH if os.path.exists(Config.DETECTION_MODEL_PATH) else None
    
    try:
        detector = YOLOv5Detector(
            model_path=model_path,
            conf_threshold=Config.YOLO_CONF_THRESHOLD,
            iou_threshold=Config.YOLO_IOU_THRESHOLD,
            image_size=Config.YOLO_IMAGE_SIZE,
            device=Config.DEVICE,
        )
    except Exception as e:
        print(f"Failed to load YOLOv5 model: {e}")
        print("Make sure you have internet connection to download the model.")
        return False
    
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    perf = PerformanceMetrics()
    
    print("\nRunning detection inference...")
    perf.start_timer()
    detections = detector.detect(test_image)
    inference_time = perf.end_timer()
    
    print(f"Inference time: {inference_time:.4f}s")
    print(f"Number of detections: {len(detections)}")
    
    if detections:
        print("\nDetections:")
        for i, det in enumerate(detections[:10]):
            print(f"  {i + 1}. {det['class_name']}: {det['confidence']:.4f} at {det['bbox']}")
    
    visual_img = draw_detections(test_image, detections)
    
    output_path = os.path.join(Config.OUTPUTS_DIR, "detection_result.jpg")
    save_image(visual_img, output_path)
    print(f"\nVisualization saved to: {output_path}")
    
    return True


def test_metrics():
    print("\n" + "=" * 50)
    print("Testing Metrics...")
    print("=" * 50)
    
    from utils import ClassificationMetrics, DetectionMetrics
    
    print("\nClassification metrics test:")
    cls_metrics = ClassificationMetrics(class_names=Config.CLASS_NAMES)
    y_true = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 2, 0, 2, 1, 0, 1, 2]
    cls_metrics.update(y_true, y_pred)
    metrics = cls_metrics.compute_metrics()
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"  Recall (macro): {metrics['recall_macro']:.4f}")
    print(f"  F1 (macro): {metrics['f1_macro']:.4f}")
    
    print("\nDetection metrics test:")
    det_metrics = DetectionMetrics(iou_threshold=0.5, class_names=["cat", "dog"])
    det_metrics.add_detections(
        "img1",
        predictions=[
            {"bbox": [10, 10, 100, 100], "confidence": 0.9, "class_id": 0},
            {"bbox": [200, 200, 300, 300], "confidence": 0.8, "class_id": 1},
        ],
        ground_truths=[
            {"bbox": [12, 12, 102, 102], "class_id": 0},
            {"bbox": [205, 205, 305, 305], "class_id": 1},
        ],
    )
    det_result = det_metrics.compute_metrics()
    print(f"  Precision: {det_result['precision']:.4f}")
    print(f"  Recall: {det_result['recall']:.4f}")
    print(f"  F1: {det_result['f1']:.4f}")
    print(f"  mAP: {det_result['map']:.4f}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Test AI Inference Service")
    parser.add_argument(
        "--test",
        choices=["all", "classification", "detection", "metrics"],
        default="all",
        help="Which test to run",
    )
    
    args = parser.parse_args()
    
    Config.ensure_dirs()
    
    success = True
    
    if args.test in ["all", "classification"]:
        success = test_classification() and success
    
    if args.test in ["all", "detection"]:
        success = test_detection() and success
    
    if args.test in ["all", "metrics"]:
        success = test_metrics() and success
    
    print("\n" + "=" * 50)
    if success:
        print("All tests passed!")
    else:
        print("Some tests failed.")
    print("=" * 50)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
