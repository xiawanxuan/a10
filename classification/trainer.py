import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

from .model import build_resnet18, get_model_params


class ClassifierTrainer:
    def __init__(
        self,
        num_classes=10,
        learning_rate=0.001,
        device="cpu",
        save_path=None,
        pretrained=True,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.save_path = save_path
        
        self.model = build_resnet18(num_classes=num_classes, pretrained=pretrained)
        self.model = self.model.to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        
        params = get_model_params(self.model, lr=learning_rate)
        self.optimizer = optim.Adam(params, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=2
        )
        
        self.best_val_acc = 0.0
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
    
    def train_epoch(self, train_loader):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(
            train_loader,
            desc="Training",
            ascii=True,
            dynamic_ncols=True,
            mininterval=0.5,
        )
        for batch_idx, (inputs, labels) in enumerate(pbar):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            avg_loss = running_loss / total
            avg_acc = correct / total
            pbar.set_postfix(
                loss=f"{avg_loss:.4f}",
                acc=f"{avg_acc:.4f}",
                refresh=True,
            )
            pbar.update(0)
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(
            val_loader,
            desc="Validating",
            ascii=True,
            dynamic_ncols=True,
            mininterval=0.5,
        )
        with torch.no_grad():
            for inputs, labels in pbar:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                avg_loss = running_loss / total
                avg_acc = correct / total
                pbar.set_postfix(
                    loss=f"{avg_loss:.4f}",
                    acc=f"{avg_acc:.4f}",
                    refresh=True,
                )
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, train_loader, val_loader, num_epochs=10):
        since = time.time()
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 40)
            
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            
            self.scheduler.step(val_loss)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
            
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                if self.save_path:
                    self.save(self.save_path)
                    print(f"Best model saved with val_acc: {val_acc:.4f}")
        
        time_elapsed = time.time() - since
        print(f"\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
        print(f"Best val Acc: {self.best_val_acc:.4f}")
        
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_accs": self.train_accs,
            "val_accs": self.val_accs,
            "best_val_acc": self.best_val_acc,
        }
    
    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "num_classes": self.num_classes,
            "best_val_acc": self.best_val_acc,
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.num_classes = checkpoint.get("num_classes", self.num_classes)
        self.best_val_acc = checkpoint.get("best_val_acc", 0.0)
        print(f"Model loaded from {path}")
