import os
import time
import threading
import uuid
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable

import torch
from torch.utils.data import DataLoader, TensorDataset

from .model import build_resnet18
from .trainer import ClassifierTrainer
from data.transforms import _to_pil_rgb, get_train_transforms, get_val_transforms
from config.settings import Config


class FineTuneStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass
class FineTuneTask:
    task_id: str
    status: FineTuneStatus = FineTuneStatus.PENDING
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    epochs: int = 5
    learning_rate: float = 1e-4
    current_epoch: int = 0
    total_epochs: int = 0
    best_val_acc: float = 0.0
    current_loss: float = 0.0
    error_message: Optional[str] = None
    num_samples: int = 0
    freeze_backbone: bool = True
    result_model_path: Optional[str] = None


class FineTuneManager:
    def __init__(self, base_model_path: str, num_classes: int = 10,
                 class_names: List[str] = None, device: str = "cpu"):
        self.base_model_path = base_model_path
        self.num_classes = num_classes
        self.class_names = class_names or [str(i) for i in range(num_classes)]
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        self._tasks: Dict[str, FineTuneTask] = {}
        self._lock = threading.Lock()
        self._current_task_thread: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()
        self._active_task_id: Optional[str] = None
        
        self._callbacks: Dict[str, List[Callable]] = {
            "on_epoch_end": [],
            "on_task_complete": [],
            "on_task_failed": [],
        }
        
        self._backup_dir = os.path.join(Config.MODELS_DIR, "backups")
        os.makedirs(self._backup_dir, exist_ok=True)
    
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
    
    def create_task(self, images: List, labels: List[int], epochs: int = 5,
                   learning_rate: float = 1e-4,
                   freeze_backbone: bool = True,
                   validation_split: float = 0.2) -> str:
        task_id = str(uuid.uuid4())
        
        task = FineTuneTask(
            task_id=task_id,
            epochs=epochs,
            total_epochs=epochs,
            learning_rate=learning_rate,
            num_samples=len(images),
            freeze_backbone=freeze_backbone,
        )
        
        with self._lock:
            self._tasks[task_id] = task
        
        self._start_task_async(task_id, images, labels, validation_split)
        
        return task_id
    
    def _start_task_async(self, task_id: str, images: List, labels: List[int],
                         validation_split: float):
        if self._current_task_thread and self._current_task_thread.is_alive():
            return
        
        self._current_task_thread = threading.Thread(
            target=self._run_fine_tune_worker,
            args=(task_id, images, labels, validation_split),
            daemon=True,
        )
        self._current_task_thread.start()
    
    def _run_fine_tune_worker(self, task_id: str, images: List, labels: List[int],
                               validation_split: float):
        task = self._tasks.get(task_id)
        if not task:
            return
        
        try:
            with self._lock:
                task.status = FineTuneStatus.RUNNING
                task.start_time = time.time()
                self._active_task_id = task_id
            
            train_loader, val_loader = self._prepare_data_loaders(
                images, labels, validation_split
            )
            
            trainer = ClassifierTrainer(
                num_classes=self.num_classes,
                learning_rate=task.learning_rate,
                device=str(self.device),
                save_path=None,
                pretrained=False,
            )
            
            if os.path.exists(self.base_model_path):
                trainer.load(self.base_model_path)
            
            if task.freeze_backbone:
                for name, param in trainer.model.named_parameters():
                    if "fc" not in name:
                        param.requires_grad = False
            
            for epoch in range(task.epochs):
                if self._stop_flag.is_set():
                    with self._lock:
                        task.status = FineTuneStatus.STOPPED
                    break
                
                train_loss, train_acc = trainer.train_epoch(train_loader)
                val_loss, val_acc = trainer.validate(val_loader)
                
                with self._lock:
                    task.current_epoch = epoch + 1
                    task.current_loss = train_loss
                    if val_acc > task.best_val_acc:
                        task.best_val_acc = val_acc
                
                self._fire_event("on_epoch_end", task_id, epoch + 1, {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                })
            
            if task.status == FineTuneStatus.RUNNING:
                model_path = os.path.join(
                    self._backup_dir, f"finetune_{task_id}.pth"
                )
                trainer.save(model_path)
                
                with self._lock:
                    task.status = FineTuneStatus.COMPLETED
                    task.result_model_path = model_path
                    task.end_time = time.time()
                
                self._fire_event("on_task_complete", task_id, model_path)
        
        except Exception as e:
            with self._lock:
                task.status = FineTuneStatus.FAILED
                task.error_message = str(e)
                task.end_time = time.time()
            self._fire_event("on_task_failed", task_id, str(e))
        
        finally:
            with self._lock:
                self._active_task_id = None
    
    def _prepare_data_loaders(self, images: List, labels: List[int],
                            validation_split: float):
        from PIL import Image
        import numpy as np
        from torch.utils.data import random_split
        
        train_transform = get_train_transforms(Config.IMAGE_SIZE)
        val_transform = get_val_transforms(Config.IMAGE_SIZE)
        
        pil_images = []
        for img in images:
            pil_img = _to_pil_rgb(img)
            pil_images.append(pil_img)
        
        total = len(pil_images)
        val_size = max(1, int(total * validation_split))
        train_size = total - val_size
        
        indices = list(range(total))
        np.random.seed(42)
        np.random.shuffle(indices)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        class _Dataset(torch.utils.data.Dataset):
            def __init__(self, images, labels, indices, transform):
                self.images = images
                self.labels = labels
                self.indices = indices
                self.transform = transform
            
            def __len__(self):
                return len(self.indices)
            
            def __getitem__(self, idx):
                real_idx = self.indices[idx]
                img = self.images[real_idx]
                label = self.labels[real_idx]
                if self.transform:
                    img = self.transform(img)
                return img, label
        
        train_dataset = _Dataset(pil_images, labels, train_indices, train_transform)
        val_dataset = _Dataset(pil_images, labels, val_indices, val_transform)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=min(Config.BATCH_SIZE, len(train_dataset)),
            shuffle=True,
            num_workers=0,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=min(Config.BATCH_SIZE, len(val_dataset)),
            shuffle=False,
            num_workers=0,
        )
        
        return train_loader, val_loader
    
    def get_task_status(self, task_id: str) -> Optional[Dict]:
        task = self._tasks.get(task_id)
        if not task:
            return None
        
        return {
            "task_id": task.task_id,
            "status": task.status.value,
            "start_time": task.start_time,
            "end_time": task.end_time,
            "current_epoch": task.current_epoch,
            "total_epochs": task.total_epochs,
            "best_val_acc": task.best_val_acc,
            "current_loss": task.current_loss,
            "error_message": task.error_message,
            "num_samples": task.num_samples,
            "freeze_backbone": task.freeze_backbone,
            "result_model_path": task.result_model_path,
        }
    
    def list_tasks(self, limit: int = 20) -> List[Dict]:
        tasks = sorted(
            self._tasks.values(),
            key=lambda t: t.start_time or 0,
            reverse=True,
        )
        return [
            {
                "task_id": t.task_id,
                "status": t.status.value,
                "start_time": t.start_time,
                "end_time": t.end_time,
                "num_samples": t.num_samples,
                "best_val_acc": t.best_val_acc,
            }
            for t in tasks[:limit]
        ]
    
    def apply_finetuned_model(self, task_id: str, inference_instance):
        task = self._tasks.get(task_id)
        if not task or task.status != FineTuneStatus.COMPLETED:
            raise ValueError(f"Task {task_id} not completed")
        
        if not task.result_model_path or not os.path.exists(task.result_model_path):
            raise ValueError("Result model not found")
        
        inference_instance._load_model(task.result_model_path)
        return True
    
    def stop_task(self, task_id: str) -> bool:
        task = self._tasks.get(task_id)
        if not task or task.status != FineTuneStatus.RUNNING:
            return False
        
        self._stop_flag.set()
        return True
    
    def is_busy(self) -> bool:
        return self._active_task_id is not None
    
    def get_active_task_id(self) -> Optional[str]:
        return self._active_task_id
