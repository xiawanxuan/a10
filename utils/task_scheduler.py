import os
import time
import uuid
import threading
import queue
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable
from collections import deque


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskType(Enum):
    CLASSIFICATION = "classification"
    DETECTION = "detection"


@dataclass
class BatchTask:
    task_id: str
    task_type: TaskType
    images: List[Any] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    priority: int = 0
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    progress: float = 0.0
    total_items: int = 0
    processed_items: int = 0
    results: List[Any] = field(default_factory=list)
    error_message: Optional[str] = None
    callback_url: Optional[str] = None
    top_k: int = 5
    conf_threshold: Optional[float] = None
    iou_threshold: Optional[float] = None


class BatchTaskScheduler:
    def __init__(self, max_workers: int = 1, max_queue_size: int = 100,
                 classification_inference=None, detection_model=None):
        self.max_workers = max_workers
        self.max_queue_size = max_queue_size
        
        self._tasks: Dict[str, BatchTask] = {}
        self._task_queue: "queue.PriorityQueue[tuple]" = queue.PriorityQueue()
        self._lock = threading.Lock()
        
        self._classification_inference = classification_inference
        self._detection_model = detection_model
        
        self._workers: List[threading.Thread] = []
        self._stop_event = threading.Event()
        
        self._recent_results: deque = deque(maxlen=100)
        
        self._callbacks: Dict[str, List[Callable]] = {
            "on_task_complete": [],
            "on_task_failed": [],
            "on_task_start": [],
            "on_progress": [],
        }
        
        self._start_workers()
    
    def register_callback(self, event: str, callback: Callable):
        if event in self._callbacks:
            self._callbacks[event].append(callback)
    
    def _fire_event(self, event: str, *args, **kwargs):
        if event in self._callbacks:
            for callback in self._callbacks[event]:
                try:
                    callback(*args, **kwargs)
                except Exception:
                    pass
    
    def _start_workers(self):
        for i in range(self.max_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                args=(i,),
                daemon=True,
                name=f"batch-worker-{i}",
            )
            worker.start()
            self._workers.append(worker)
    
    def _worker_loop(self, worker_id: int):
        while not self._stop_event.is_set():
            try:
                item = self._task_queue.get(timeout=1.0)
                if item is None:
                    continue
                
                _, task_id = item
                task = self._tasks.get(task_id)
                if not task or task.status != TaskStatus.PENDING:
                    continue
                
                self._execute_task(task)
                
            except queue.Empty:
                continue
            except Exception:
                continue
    
    def _execute_task(self, task: BatchTask):
        with self._lock:
            task.status = TaskStatus.RUNNING
            task.started_at = time.time()
        
        self._fire_event("on_task_start", task.task_id)
        
        try:
            results = []
            total = len(task.images)
            task.total_items = total
            
            for i, image in enumerate(task.images):
                if task.status == TaskStatus.CANCELLED:
                    break
                
                if task.task_type == TaskType.CLASSIFICATION:
                    result = self._do_classification(image, task)
                elif task.task_type == TaskType.DETECTION:
                    result = self._do_detection(image, task)
                else:
                    raise ValueError(f"Unknown task type: {task.task_type}")
                
                results.append({
                    "index": i,
                    "result": result,
                })
                
                task.processed_items = i + 1
                task.progress = (i + 1) / total if total > 0 else 1.0
                
                if i % max(1, total // 10) == 0 or i == total - 1:
                    self._fire_event("on_progress", task.task_id, task.progress)
            
            task.results = results
            
            with self._lock:
                task.status = TaskStatus.COMPLETED
                task.completed_at = time.time()
            
            self._fire_event("on_task_complete", task.task_id, results)
            
            self._recent_results.append({
                "task_id": task.task_id,
                "task_type": task.task_type.value,
                "completed_at": task.completed_at,
                "num_images": total,
            })
            
        except Exception as e:
            with self._lock:
                task.status = TaskStatus.FAILED
                task.error_message = str(e)
                task.completed_at = time.time()
            self._fire_event("on_task_failed", task.task_id, str(e))
    
    def _do_classification(self, image, task: BatchTask):
        if not self._classification_inference:
            raise ValueError("Classification model not initialized")
        
        return self._classification_inference.predict(image, top_k=task.top_k)
    
    def _do_detection(self, image, task: BatchTask):
        if not self._detection_model:
            raise ValueError("Detection model not initialized")
        
        if task.conf_threshold is not None:
            self._detection_model.set_conf_threshold(task.conf_threshold)
        if task.iou_threshold is not None:
            self._detection_model.set_iou_threshold(task.iou_threshold)
        
        return self._detection_model.detect(image)
    
    def submit_classification_task(self, images: List, top_k: int = 5,
                                   priority: int = 0,
                                   callback_url: str = None) -> str:
        task_id = str(uuid.uuid4())
        task = BatchTask(
            task_id=task_id,
            task_type=TaskType.CLASSIFICATION,
            images=images,
            priority=priority,
            top_k=top_k,
            callback_url=callback_url,
        )
        
        return self._submit_task(task)
    
    def submit_detection_task(self, images: List,
                              conf_threshold: float = None,
                              iou_threshold: float = None,
                              priority: int = 0,
                              callback_url: str = None) -> str:
        task_id = str(uuid.uuid4())
        task = BatchTask(
            task_id=task_id,
            task_type=TaskType.DETECTION,
            images=images,
            priority=priority,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            callback_url=callback_url,
        )
        
        return self._submit_task(task)
    
    def _submit_task(self, task: BatchTask) -> str:
        if self._task_queue.qsize() >= self.max_queue_size:
            raise RuntimeError("Task queue is full")
        
        with self._lock:
            self._tasks[task.task_id] = task
        
        self._task_queue.put((-task.priority, task.task_id))
        
        return task.task_id
    
    def get_task_status(self, task_id: str) -> Optional[Dict]:
        task = self._tasks.get(task_id)
        if not task:
            return None
        
        return {
            "task_id": task.task_id,
            "task_type": task.task_type.value,
            "status": task.status.value,
            "priority": task.priority,
            "created_at": task.created_at,
            "started_at": task.started_at,
            "completed_at": task.completed_at,
            "progress": task.progress,
            "total_items": task.total_items,
            "processed_items": task.processed_items,
            "error_message": task.error_message,
            "has_results": task.status == TaskStatus.COMPLETED,
        }
    
    def get_task_results(self, task_id: str) -> Optional[Dict]:
        task = self._tasks.get(task_id)
        if not task:
            return None
        
        return {
            "task_id": task.task_id,
            "task_type": task.task_type.value,
            "status": task.status.value,
            "results": task.results,
            "total_items": task.total_items,
            "completed_at": task.completed_at,
        }
    
    def cancel_task(self, task_id: str) -> bool:
        task = self._tasks.get(task_id)
        if not task:
            return False
        
        if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
            return False
        
        with self._lock:
            task.status = TaskStatus.CANCELLED
            task.completed_at = time.time()
        
        return True
    
    def list_tasks(self, status: str = None, limit: int = 20) -> List[Dict]:
        tasks = list(self._tasks.values())
        tasks.sort(key=lambda t: t.created_at, reverse=True)
        
        if status:
            status_enum = TaskStatus(status)
            tasks = [t for t in tasks if t.status == status_enum]
        
        return [
            {
                "task_id": t.task_id,
                "task_type": t.task_type.value,
                "status": t.status.value,
                "priority": t.priority,
                "created_at": t.created_at,
                "progress": t.progress,
                "total_items": t.total_items,
            }
            for t in tasks[:limit]
        ]
    
    def get_queue_stats(self) -> Dict:
        pending_count = sum(
            1 for t in self._tasks.values() if t.status == TaskStatus.PENDING
        )
        running_count = sum(
            1 for t in self._tasks.values() if t.status == TaskStatus.RUNNING
        )
        completed_count = sum(
            1 for t in self._tasks.values() if t.status == TaskStatus.COMPLETED
        )
        failed_count = sum(
            1 for t in self._tasks.values() if t.status == TaskStatus.FAILED
        )
        
        return {
            "queue_size": self._task_queue.qsize(),
            "max_queue_size": self.max_queue_size,
            "workers": self.max_workers,
            "pending": pending_count,
            "running": running_count,
            "completed": completed_count,
            "failed": failed_count,
            "recent_completed": list(self._recent_results),
        }
    
    def set_classification_inference(self, inference_instance):
        self._classification_inference = inference_instance
    
    def set_detection_model(self, detection_model):
        self._detection_model = detection_model
    
    def shutdown(self):
        self._stop_event.set()
        for worker in self._workers:
            worker.join(timeout=5.0)
    
    def clear_old_tasks(self, older_than_seconds: int = 3600):
        now = time.time()
        with self._lock:
            to_remove = []
            for task_id, task in self._tasks.items():
                if task.completed_at and (now - task.completed_at) > older_than_seconds:
                    if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                        to_remove.append(task_id)
            
            for task_id in to_remove:
                del self._tasks[task_id]
        
        return len(to_remove)
