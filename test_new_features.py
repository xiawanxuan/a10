import os
import sys
import time
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import Config
from classification import ClassifierInference, FineTuneManager
from utils import (
    BatchTaskScheduler,
    TaskStatus,
    InferenceStorage,
)


def test_storage():
    print("=" * 60)
    print("Testing Inference Storage...")
    print("=" * 60)
    
    Config.ensure_dirs()
    
    storage = InferenceStorage()
    
    print("\n1. Saving classification results...")
    for i in range(10):
        results = [
            {"class_id": i % 10, "class_name": Config.CLASS_NAMES[i % 10], "confidence": 0.8 + i * 0.01},
            {"class_id": (i + 1) % 10, "class_name": Config.CLASS_NAMES[(i + 1) % 10], "confidence": 0.1},
        ]
        result_id = storage.save_classification_result(
            image_id=f"test_img_{i}",
            results=results,
            inference_time=0.1 + i * 0.01,
        )
        print(f"   Saved: {result_id}")
    
    print("\n2. Saving detection results...")
    for i in range(5):
        detections = [
            {"bbox": [10, 10, 100, 100], "confidence": 0.9, "class_id": 0, "class_name": "person"},
            {"bbox": [200, 200, 300, 300], "confidence": 0.8, "class_id": 1, "class_name": "car"},
        ]
        result_id = storage.save_detection_result(
            image_id=f"test_det_{i}",
            detections=detections,
            inference_time=0.2 + i * 0.01,
        )
        print(f"   Saved: {result_id}")
    
    print("\n3. Listing classification results...")
    results = storage.list_classification_results(limit=5)
    print(f"   Got {len(results)} results")
    if results:
        r = results[0]
        print(f"   Latest: id={r['id']}, class={r['top_class_name']}, conf={r['top_confidence']:.3f}")
    
    print("\n4. Getting classification stats...")
    stats = storage.get_classification_stats()
    print(f"   Total: {stats['total_count']}")
    print(f"   Avg time: {stats['avg_inference_time']:.4f}s")
    print(f"   Avg confidence: {stats['avg_confidence']:.4f}")
    print(f"   Class distribution: {len(stats['class_distribution'])} classes")
    
    print("\n5. Getting detection stats...")
    det_stats = storage.get_detection_stats()
    print(f"   Total: {det_stats['total_count']}")
    print(f"   Avg detections: {det_stats['avg_detections_per_image']:.1f}")
    
    print("\n[PASS] Inference Storage test passed!")
    return True


def test_batch_scheduler():
    print("\n" + "=" * 60)
    print("Testing Batch Task Scheduler...")
    print("=" * 60)
    
    model_path = Config.CLASSIFICATION_MODEL_PATH
    if not os.path.exists(model_path):
        print("   [SKIP] Classification model not found, skipping batch test")
        return True
    
    classifier = ClassifierInference(
        model_path=model_path,
        num_classes=len(Config.CLASS_NAMES),
        class_names=Config.CLASS_NAMES,
        device=Config.DEVICE,
    )
    
    scheduler = BatchTaskScheduler(
        max_workers=1,
        max_queue_size=50,
        classification_inference=classifier,
        detection_model=None,
    )
    
    print("\n1. Creating test images...")
    test_images = []
    for i in range(5):
        img_array = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        test_images.append(img)
    print(f"   Created {len(test_images)} test images")
    
    print("\n2. Submitting classification batch task...")
    task_id = scheduler.submit_classification_task(
        images=test_images,
        top_k=3,
        priority=0,
    )
    print(f"   Task submitted: {task_id}")
    
    print("\n3. Checking task status...")
    time.sleep(0.5)
    status = scheduler.get_task_status(task_id)
    print(f"   Status: {status['status']}")
    print(f"   Progress: {status['progress']:.0%}")
    print(f"   Processed: {status['processed_items']}/{status['total_items']}")
    
    print("\n4. Waiting for task to complete...")
    max_wait = 30
    waited = 0
    while waited < max_wait:
        status = scheduler.get_task_status(task_id)
        if status["status"] in ["completed", "failed"]:
            break
        time.sleep(0.5)
        waited += 0.5
    
    final_status = scheduler.get_task_status(task_id)
    print(f"   Final status: {final_status['status']}")
    
    if final_status["status"] == "completed":
        print("\n5. Getting task results...")
        results = scheduler.get_task_results(task_id)
        print(f"   Got {len(results['results'])} results")
        if results["results"]:
            first_result = results["results"][0]["result"]
            print(f"   First image top class: {first_result[0]['class_name']}")
    
    print("\n6. Getting queue stats...")
    stats = scheduler.get_queue_stats()
    print(f"   Pending: {stats['pending']}")
    print(f"   Running: {stats['running']}")
    print(f"   Completed: {stats['completed']}")
    
    print("\n7. Listing tasks...")
    tasks = scheduler.list_tasks(limit=5)
    print(f"   Listed {len(tasks)} tasks")
    
    scheduler.shutdown()
    
    print("\n[PASS] Batch Task Scheduler test passed!")
    return True


def test_fine_tuning():
    print("\n" + "=" * 60)
    print("Testing Fine-tuning Manager...")
    print("=" * 60)
    
    model_path = Config.CLASSIFICATION_MODEL_PATH
    if not os.path.exists(model_path):
        print("   [SKIP] Classification model not found, skipping fine-tune test")
        return True
    
    ft_manager = FineTuneManager(
        base_model_path=model_path,
        num_classes=len(Config.CLASS_NAMES),
        class_names=Config.CLASS_NAMES,
        device=Config.DEVICE,
    )
    
    print("\n1. Creating fine-tuning samples...")
    images = []
    labels = []
    for i in range(20):
        img_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        images.append(img)
        labels.append(i % 10)
    print(f"   Created {len(images)} training samples")
    
    print("\n2. Starting fine-tuning task...")
    task_id = ft_manager.create_task(
        images=images,
        labels=labels,
        epochs=2,
        learning_rate=1e-4,
        freeze_backbone=True,
    )
    print(f"   Task started: {task_id}")
    
    print("\n3. Checking task status...")
    time.sleep(1.0)
    status = ft_manager.get_task_status(task_id)
    print(f"   Status: {status['status']}")
    if status["status"] == "running":
        print(f"   Epoch: {status['current_epoch']}/{status['total_epochs']}")
    
    print("\n4. Waiting for fine-tuning to complete...")
    max_wait = 120
    waited = 0
    while waited < max_wait:
        status = ft_manager.get_task_status(task_id)
        if status["status"] in ["completed", "failed", "stopped"]:
            break
        time.sleep(2)
        waited += 2
        print(f"   ... waited {waited}s, status: {status['status']}")
    
    final_status = ft_manager.get_task_status(task_id)
    print(f"\n   Final status: {final_status['status']}")
    if final_status["status"] == "completed":
        print(f"   Best val acc: {final_status['best_val_acc']:.4f}")
        print(f"   Result model: {final_status['result_model_path']}")
    
    print("\n5. Listing fine-tune tasks...")
    tasks = ft_manager.list_tasks(limit=5)
    print(f"   Listed {len(tasks)} tasks")
    
    print("\n[PASS] Fine-tuning Manager test passed!")
    return True


def main():
    print("\n" + "=" * 60)
    print("NEW FEATURES TEST SUITE")
    print("=" * 60)
    
    all_passed = True
    
    try:
        test_storage()
    except Exception as e:
        print(f"\n[FAIL] Storage test failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    try:
        test_batch_scheduler()
    except Exception as e:
        print(f"\n[FAIL] Batch scheduler test failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    try:
        test_fine_tuning()
    except Exception as e:
        print(f"\n[FAIL] Fine-tuning test failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("ALL NEW FEATURE TESTS PASSED!")
    else:
        print("SOME TESTS FAILED")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
