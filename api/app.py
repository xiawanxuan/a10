import os
import io
import base64
import uuid
import time
import numpy as np
from PIL import Image
import cv2
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

from config.settings import Config
from classification import ClassifierInference, FineTuneManager
from detection import YOLOv5Detector
from utils import (
    draw_detections,
    draw_classification,
    PerformanceMetrics,
    BatchTaskScheduler,
    InferenceStorage,
)


def create_app():
    app = Flask(__name__)
    CORS(app)
    
    Config.ensure_dirs()
    
    classifier = None
    detector = None
    perf_metrics = PerformanceMetrics()
    fine_tune_manager = None
    task_scheduler = None
    storage = None
    
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
    
    def get_fine_tune_manager():
        nonlocal fine_tune_manager
        if fine_tune_manager is None:
            fine_tune_manager = FineTuneManager(
                base_model_path=Config.CLASSIFICATION_MODEL_PATH,
                num_classes=len(Config.CLASS_NAMES),
                class_names=Config.CLASS_NAMES,
                device=Config.DEVICE,
            )
        return fine_tune_manager
    
    def get_task_scheduler():
        nonlocal task_scheduler
        if task_scheduler is None:
            task_scheduler = BatchTaskScheduler(
                max_workers=1,
                max_queue_size=100,
                classification_inference=get_classifier(),
                detection_model=get_detector(),
            )
        else:
            if classifier is not None:
                task_scheduler.set_classification_inference(classifier)
            if detector is not None:
                task_scheduler.set_detection_model(detector)
        return task_scheduler
    
    def get_storage():
        nonlocal storage
        if storage is None:
            storage = InferenceStorage()
        return storage
    
    def decode_image_from_request():
        if "image" in request.files:
            file = request.files["image"]
            image_bytes = file.read()
        elif request.is_json and "image_base64" in request.json:
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
    
    def decode_images_from_batch_request():
        images = []
        
        if "images" in request.files:
            files = request.files.getlist("images")
            for file in files:
                try:
                    img = Image.open(io.BytesIO(file.read())).convert("RGB")
                    images.append(img)
                except Exception:
                    pass
        elif request.is_json and "images_base64" in request.json:
            for img_b64 in request.json["images_base64"]:
                try:
                    data = img_b64
                    if data.startswith("data:image"):
                        data = data.split(",")[1]
                    img_bytes = base64.b64decode(data)
                    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    images.append(img)
                except Exception:
                    pass
        
        return images
    
    @app.route("/", methods=["GET"])
    def index():
        return jsonify({
            "service": "AI Inference Service",
            "version": "2.0.0",
            "endpoints": {
                "classification": {
                    "predict": "/api/v1/classification/predict",
                    "predict_visual": "/api/v1/classification/predict/visual",
                    "finetune": "/api/v1/classification/finetune",
                    "finetune_status": "/api/v1/classification/finetune/<task_id>",
                    "finetune_list": "/api/v1/classification/finetune/list",
                    "finetune_apply": "/api/v1/classification/finetune/<task_id>/apply",
                },
                "detection": {
                    "detect": "/api/v1/detection/detect",
                    "detect_visual": "/api/v1/detection/detect/visual",
                },
                "batch": {
                    "classification": "/api/v1/batch/classification",
                    "detection": "/api/v1/batch/detection",
                    "status": "/api/v1/batch/<task_id>",
                    "results": "/api/v1/batch/<task_id>/results",
                    "cancel": "/api/v1/batch/<task_id>/cancel",
                    "list": "/api/v1/batch/list",
                    "stats": "/api/v1/batch/stats",
                },
                "storage": {
                    "classification_list": "/api/v1/storage/classification",
                    "classification_detail": "/api/v1/storage/classification/<result_id>",
                    "detection_list": "/api/v1/storage/detection",
                    "detection_detail": "/api/v1/storage/detection/<result_id>",
                    "stats": "/api/v1/storage/stats",
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
        ft_manager = get_fine_tune_manager()
        scheduler = get_task_scheduler()
        storage = get_storage()
        
        return jsonify({
            "classification_available": classifier is not None,
            "detection_available": detector is not None,
            "fine_tune_busy": ft_manager.is_busy(),
            "performance": perf_metrics.get_inference_stats(),
            "batch_stats": scheduler.get_queue_stats(),
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
        save_result = request.args.get("save", "true").lower() == "true"
        image_id = request.args.get("image_id", None)
        
        perf_metrics.start_timer()
        results = classifier.predict(image, top_k=top_k)
        inference_time = perf_metrics.end_timer()
        
        result_id = None
        if save_result:
            storage = get_storage()
            result_id = storage.save_classification_result(
                image_path=None,
                image_id=image_id,
                results=results,
                inference_time=inference_time,
                model_version="v1",
            )
        
        return jsonify({
            "success": True,
            "inference_time": inference_time,
            "results": results,
            "result_id": result_id,
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
        save_result = request.args.get("save", "true").lower() == "true"
        image_id = request.args.get("image_id", None)
        
        if conf_threshold is not None:
            detector.set_conf_threshold(conf_threshold)
        if iou_threshold is not None:
            detector.set_iou_threshold(iou_threshold)
        
        perf_metrics.start_timer()
        detections = detector.detect(image)
        inference_time = perf_metrics.end_timer()
        
        result_id = None
        if save_result:
            storage = get_storage()
            result_id = storage.save_detection_result(
                image_path=None,
                image_id=image_id,
                detections=detections,
                inference_time=inference_time,
                conf_threshold=detector.conf_threshold,
                iou_threshold=detector.iou_threshold,
                model_version="v1",
            )
        
        return jsonify({
            "success": True,
            "inference_time": inference_time,
            "num_detections": len(detections),
            "detections": detections,
            "result_id": result_id,
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
    
    @app.route("/api/v1/classification/finetune", methods=["POST"])
    def classification_finetune():
        classifier = get_classifier()
        if classifier is None:
            return jsonify({"error": "Classification model not available"}), 503
        
        ft_manager = get_fine_tune_manager()
        
        data = request.json if request.is_json else {}
        
        epochs = data.get("epochs", 5)
        learning_rate = data.get("learning_rate", 1e-4)
        freeze_backbone = data.get("freeze_backbone", True)
        
        images_data = data.get("images", [])
        labels = data.get("labels", [])
        
        if not images_data or not labels:
            return jsonify({"error": "Images and labels are required"}), 400
        
        if len(images_data) != len(labels):
            return jsonify({"error": "Number of images and labels must match"}), 400
        
        images = []
        for img_data in images_data:
            try:
                if img_data.startswith("data:image"):
                    img_data = img_data.split(",")[1]
                img_bytes = base64.b64decode(img_data)
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                images.append(img)
            except Exception:
                pass
        
        if not images:
            return jsonify({"error": "No valid images provided"}), 400
        
        try:
            task_id = ft_manager.create_task(
                images=images,
                labels=labels,
                epochs=epochs,
                learning_rate=learning_rate,
                freeze_backbone=freeze_backbone,
            )
            return jsonify({
                "success": True,
                "task_id": task_id,
                "message": "Fine-tuning task started",
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route("/api/v1/classification/finetune/<task_id>", methods=["GET"])
    def classification_finetune_status(task_id):
        ft_manager = get_fine_tune_manager()
        status = ft_manager.get_task_status(task_id)
        
        if status is None:
            return jsonify({"error": "Task not found"}), 404
        
        return jsonify({
            "success": True,
            "task": status,
        })
    
    @app.route("/api/v1/classification/finetune/list", methods=["GET"])
    def classification_finetune_list():
        ft_manager = get_fine_tune_manager()
        limit = request.args.get("limit", 20, type=int)
        tasks = ft_manager.list_tasks(limit=limit)
        
        return jsonify({
            "success": True,
            "tasks": tasks,
        })
    
    @app.route("/api/v1/classification/finetune/<task_id>/apply", methods=["POST"])
    def classification_finetune_apply(task_id):
        classifier = get_classifier()
        if classifier is None:
            return jsonify({"error": "Classification model not available"}), 503
        
        ft_manager = get_fine_tune_manager()
        
        try:
            ft_manager.apply_finetuned_model(task_id, classifier)
            return jsonify({
                "success": True,
                "message": "Finetuned model applied successfully",
            })
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route("/api/v1/batch/classification", methods=["POST"])
    def batch_classification():
        classifier = get_classifier()
        if classifier is None:
            return jsonify({"error": "Classification model not available"}), 503
        
        scheduler = get_task_scheduler()
        
        top_k = request.args.get("top_k", 5, type=int)
        priority = request.args.get("priority", 0, type=int)
        
        images = decode_images_from_batch_request()
        
        if not images:
            return jsonify({"error": "No valid images provided"}), 400
        
        try:
            task_id = scheduler.submit_classification_task(
                images=images,
                top_k=top_k,
                priority=priority,
            )
            return jsonify({
                "success": True,
                "task_id": task_id,
                "num_images": len(images),
                "message": "Batch classification task submitted",
            })
        except RuntimeError as e:
            return jsonify({"error": str(e)}), 503
    
    @app.route("/api/v1/batch/detection", methods=["POST"])
    def batch_detection():
        detector = get_detector()
        if detector is None:
            return jsonify({"error": "Detection model not available"}), 503
        
        scheduler = get_task_scheduler()
        
        conf_threshold = request.args.get("conf_threshold", None, type=float)
        iou_threshold = request.args.get("iou_threshold", None, type=float)
        priority = request.args.get("priority", 0, type=int)
        
        images = decode_images_from_batch_request()
        
        if not images:
            return jsonify({"error": "No valid images provided"}), 400
        
        try:
            task_id = scheduler.submit_detection_task(
                images=images,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                priority=priority,
            )
            return jsonify({
                "success": True,
                "task_id": task_id,
                "num_images": len(images),
                "message": "Batch detection task submitted",
            })
        except RuntimeError as e:
            return jsonify({"error": str(e)}), 503
    
    @app.route("/api/v1/batch/<task_id>", methods=["GET"])
    def batch_task_status(task_id):
        scheduler = get_task_scheduler()
        status = scheduler.get_task_status(task_id)
        
        if status is None:
            return jsonify({"error": "Task not found"}), 404
        
        return jsonify({
            "success": True,
            "task": status,
        })
    
    @app.route("/api/v1/batch/<task_id>/results", methods=["GET"])
    def batch_task_results(task_id):
        scheduler = get_task_scheduler()
        results = scheduler.get_task_results(task_id)
        
        if results is None:
            return jsonify({"error": "Task not found"}), 404
        
        return jsonify({
            "success": True,
            "task": results,
        })
    
    @app.route("/api/v1/batch/<task_id>/cancel", methods=["POST"])
    def batch_task_cancel(task_id):
        scheduler = get_task_scheduler()
        success = scheduler.cancel_task(task_id)
        
        if not success:
            return jsonify({"error": "Task not found or already completed"}), 400
        
        return jsonify({
            "success": True,
            "message": "Task cancelled",
        })
    
    @app.route("/api/v1/batch/list", methods=["GET"])
    def batch_task_list():
        scheduler = get_task_scheduler()
        status = request.args.get("status", None)
        limit = request.args.get("limit", 20, type=int)
        
        tasks = scheduler.list_tasks(status=status, limit=limit)
        
        return jsonify({
            "success": True,
            "tasks": tasks,
        })
    
    @app.route("/api/v1/batch/stats", methods=["GET"])
    def batch_task_stats():
        scheduler = get_task_scheduler()
        stats = scheduler.get_queue_stats()
        
        return jsonify({
            "success": True,
            "stats": stats,
        })
    
    @app.route("/api/v1/storage/classification", methods=["GET"])
    def storage_classification_list():
        storage = get_storage()
        
        class_name = request.args.get("class_name", None)
        min_confidence = request.args.get("min_confidence", None, type=float)
        limit = request.args.get("limit", 50, type=int)
        offset = request.args.get("offset", 0, type=int)
        
        results = storage.list_classification_results(
            class_name=class_name,
            min_confidence=min_confidence,
            limit=limit,
            offset=offset,
        )
        
        return jsonify({
            "success": True,
            "results": results,
            "total": len(results),
        })
    
    @app.route("/api/v1/storage/classification/<result_id>", methods=["GET"])
    def storage_classification_detail(result_id):
        storage = get_storage()
        result = storage.get_classification_result(result_id)
        
        if result is None:
            return jsonify({"error": "Result not found"}), 404
        
        return jsonify({
            "success": True,
            "result": result,
        })
    
    @app.route("/api/v1/storage/detection", methods=["GET"])
    def storage_detection_list():
        storage = get_storage()
        
        class_name = request.args.get("class_name", None)
        min_confidence = request.args.get("min_confidence", None, type=float)
        limit = request.args.get("limit", 50, type=int)
        offset = request.args.get("offset", 0, type=int)
        
        results = storage.list_detection_results(
            class_name=class_name,
            min_confidence=min_confidence,
            limit=limit,
            offset=offset,
        )
        
        return jsonify({
            "success": True,
            "results": results,
            "total": len(results),
        })
    
    @app.route("/api/v1/storage/detection/<result_id>", methods=["GET"])
    def storage_detection_detail(result_id):
        storage = get_storage()
        result = storage.get_detection_result(result_id)
        
        if result is None:
            return jsonify({"error": "Result not found"}), 404
        
        return jsonify({
            "success": True,
            "result": result,
        })
    
    @app.route("/api/v1/storage/stats", methods=["GET"])
    def storage_stats():
        storage = get_storage()
        
        days = request.args.get("days", 7, type=int)
        start_time = time.time() - days * 86400
        
        cls_stats = storage.get_classification_stats(start_time=start_time)
        det_stats = storage.get_detection_stats(start_time=start_time)
        
        return jsonify({
            "success": True,
            "classification": cls_stats,
            "detection": det_stats,
            "days": days,
        })
    
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
