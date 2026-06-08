import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Config:
    BASE_DIR = BASE_DIR
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
    DATA_DIR = os.path.join(BASE_DIR, "data")
    
    CLASSIFICATION_MODEL_PATH = os.path.join(MODELS_DIR, "resnet18_classifier.pth")
    DETECTION_MODEL_PATH = os.path.join(MODELS_DIR, "yolov5n.pt")
    
    CLASS_NAMES = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]
    
    IMAGE_SIZE = 224
    BATCH_SIZE = 32
    NUM_WORKERS = 2
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 10
    DEVICE = "cpu"
    
    YOLO_CONF_THRESHOLD = 0.25
    YOLO_IOU_THRESHOLD = 0.45
    YOLO_IMAGE_SIZE = 640
    
    API_HOST = "0.0.0.0"
    API_PORT = 5000
    API_DEBUG = False
    
    @classmethod
    def ensure_dirs(cls):
        for dir_path in [cls.MODELS_DIR, cls.OUTPUTS_DIR, cls.DATA_DIR]:
            os.makedirs(dir_path, exist_ok=True)
