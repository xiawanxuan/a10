import os
import io
import base64
import numpy as np
from PIL import Image
import cv2
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

from config.settings import Config
from classification import ClassifierInference
from detection import YOLOv5Detector
from utils import (
    draw_detections,
    draw_classification,
    PerformanceMetrics,
)


def create_app():
    app = Flask(__name__)
    CORS(app)
    
    Config.ensure_dirs()
    
    classifier = None
    detector = None
    perf_metrics = PerformanceMetrics()
    
    def get_classifier():
        nonlocal classifier
        if classifier is None:
            if os.path.exists(Config.CLASSIFICATION_MODEL_PATH):
                classifier = ClassifierInference(
                    model_path=Config.CLASSIFICATION_MODEL_PATH,
                    num_classes=len(Config.CLASS_NAMES),
                    class_names=Config.CLASS_NAMES,
                    device=Config.DEVICE,
                )
            else:
                print(f"Warning: Classification model not found at {Config.CLASSIFICATION_MODEL_PATH}")
                print("Classification API will not be available.")
        return classifier
    
    def get_detector():
        nonlocal detector
        if detector is None:
            try:
                model_path = Config.DETECTION_MODEL_PATH if os.path.exists(Config.DETECTION_MODEL_PATH) else None
                detector = YOLOv5Detector(
                    model_path=model_path,
                    conf_threshold=Config.YOLO_CONF_THRESHOLD,
                    iou_threshold=Config.YOLO_IOU_THRESHOLD,
                    image_size=Config.YOLO_IMAGE_SIZE,
                    device=Config.DEVICE,
                )
            except Exception as e:
                print(f"Warning: Failed to load YOLOv5 model: {e}")
                print("Detection API may not work properly.")
        return detector
    
    def decode_image_from_request():
        if "image" in request.files:
            file = request.files["image"]
            image_bytes = file.read()
        elif "image_base64" in request.json:
            image_data = request.json["image_base64"]
            if image_data.startswith("data:image"):
                image_data = image_data.split(",")[1]
            image_bytes = base64.b64decode(image_data)
        else:
            return None, "No image provided"
        
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            return image, None
        except Exception as e:
            return None, str(e)
    
    @app.route("/", methods=["GET"])
    def index():
        return jsonify({
            "service": "AI Inference Service",
            "version": "1.0.0",
            "endpoints": {
                "classification": {
                    "predict": "/api/v1/classification/predict",
                    "predict_visual": "/api/v1/classification/predict/visual",
                },
                "detection": {
                    "detect": "/api/v1/detection/detect",
                    "detect_visual": "/api/v1/detection/detect/visual",
                },
                "health": "/health",
                "status": "/api/v1/status",
            },
        })
    
    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok"})
    
    @app.route("/api/v1/status", methods=["GET"])
    def status():
        classifier = get_classifier()
        detector = get_detector()
        
        return jsonify({
            "classification_available": classifier is not None,
            "detection_available": detector is not None,
            "performance": perf_metrics.get_inference_stats(),
            "config": {
                "device": Config.DEVICE,
                "image_size": Config.IMAGE_SIZE,
                "yolo_conf_threshold": Config.YOLO_CONF_THRESHOLD,
                "yolo_iou_threshold": Config.YOLO_IOU_THRESHOLD,
                "class_names": Config.CLASS_NAMES,
            },
        })
    
    @app.route("/api/v1/classification/predict", methods=["POST"])
    def classification_predict():
        classifier = get_classifier()
        if classifier is None:
            return jsonify({"error": "Classification model not available"}), 503
        
        image, error = decode_image_from_request()
        if error:
            return jsonify({"error": error}), 400
        
        top_k = request.args.get("top_k", 5, type=int)
        top_k = min(top_k, len(Config.CLASS_NAMES))
        
        perf_metrics.start_timer()
        results = classifier.predict(image, top_k=top_k)
        inference_time = perf_metrics.end_timer()
        
        return jsonify({
            "success": True,
            "inference_time": inference_time,
            "results": results,
        })
    
    @app.route("/api/v1/classification/predict/visual", methods=["POST"])
    def classification_predict_visual():
        classifier = get_classifier()
        if classifier is None:
            return jsonify({"error": "Classification model not available"}), 503
        
        image, error = decode_image_from_request()
        if error:
            return jsonify({"error": error}), 400
        
        top_k = request.args.get("top_k", 5, type=int)
        top_k = min(top_k, len(Config.CLASS_NAMES))
        
        perf_metrics.start_timer()
        results = classifier.predict(image, top_k=top_k)
        inference_time = perf_metrics.end_timer()
        
        img_array = np.array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        visual_img = draw_classification(img_bgr, results, top_k=top_k)
        
        _, buffer = cv2.imencode(".jpg", visual_img)
        img_bytes = io.BytesIO(buffer.tobytes())
        
        return send_file(
            img_bytes,
            mimetype="image/jpeg",
            as_attachment=False,
        )
    
    @app.route("/api/v1/detection/detect", methods=["POST"])
    def detection_detect():
        detector = get_detector()
        if detector is None:
            return jsonify({"error": "Detection model not available"}), 503
        
        image, error = decode_image_from_request()
        if error:
            return jsonify({"error": error}), 400
        
        conf_threshold = request.args.get("conf_threshold", None, type=float)
        iou_threshold = request.args.get("iou_threshold", None, type=float)
        
        if conf_threshold is not None:
            detector.set_conf_threshold(conf_threshold)
        if iou_threshold is not None:
            detector.set_iou_threshold(iou_threshold)
        
        perf_metrics.start_timer()
        detections = detector.detect(image)
        inference_time = perf_metrics.end_timer()
        
        return jsonify({
            "success": True,
            "inference_time": inference_time,
            "num_detections": len(detections),
            "detections": detections,
        })
    
    @app.route("/api/v1/detection/detect/visual", methods=["POST"])
    def detection_detect_visual():
        detector = get_detector()
        if detector is None:
            return jsonify({"error": "Detection model not available"}), 503
        
        image, error = decode_image_from_request()
        if error:
            return jsonify({"error": error}), 400
        
        conf_threshold = request.args.get("conf_threshold", None, type=float)
        iou_threshold = request.args.get("iou_threshold", None, type=float)
        
        if conf_threshold is not None:
            detector.set_conf_threshold(conf_threshold)
        if iou_threshold is not None:
            detector.set_iou_threshold(iou_threshold)
        
        perf_metrics.start_timer()
        detections = detector.detect(image)
        inference_time = perf_metrics.end_timer()
        
        img_array = np.array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        visual_img = draw_detections(img_bgr, detections)
        
        _, buffer = cv2.imencode(".jpg", visual_img)
        img_bytes = io.BytesIO(buffer.tobytes())
        
        return send_file(
            img_bytes,
            mimetype="image/jpeg",
            as_attachment=False,
        )
    
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({"error": "Not found"}), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({"error": "Internal server error"}), 500
    
    return app


def run_app(host=None, port=None, debug=None):
    host = host or Config.API_HOST
    port = port or Config.API_PORT
    debug = debug if debug is not None else Config.API_DEBUG
    
    app = create_app()
    print(f"Starting AI Inference Service on http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    run_app()
