import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets


class ImageFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}
        
        if os.path.isdir(root_dir):
            classes = sorted([d for d in os.listdir(root_dir) 
                            if os.path.isdir(os.path.join(root_dir, d))])
            self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
            
            for cls_name in classes:
                cls_dir = os.path.join(root_dir, cls_name)
                for img_name in os.listdir(cls_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        self.image_paths.append(os.path.join(cls_dir, img_name))
                        self.labels.append(self.class_to_idx[cls_name])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def create_data_loaders(
    train_dir=None,
    val_dir=None,
    test_dir=None,
    batch_size=32,
    num_workers=2,
    image_size=224,
    use_cifar10=False
):
    from .transforms import get_train_transforms, get_val_transforms, get_test_transforms
    
    loaders = {}
    
    if use_cifar10:
        from config.settings import Config
        data_dir = os.path.join(Config.DATA_DIR, "cifar10")
        
        train_transform = get_train_transforms(image_size)
        test_transform = get_test_transforms(image_size)
        
        train_dataset = datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=train_transform
        )
        test_dataset = datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=test_transform
        )
        
        val_size = int(len(train_dataset) * 0.1)
        train_size = len(train_dataset) - val_size
        
        from torch.utils.data import random_split
        train_dataset, val_dataset = random_split(
            train_dataset, [train_size, val_size]
        )
        
        val_dataset.dataset.transform = test_transform
        
        loaders["train"] = DataLoader(
            train_dataset, batch_size=batch_size,
            shuffle=True, num_workers=num_workers
        )
        loaders["val"] = DataLoader(
            val_dataset, batch_size=batch_size,
            shuffle=False, num_workers=num_workers
        )
        loaders["test"] = DataLoader(
            test_dataset, batch_size=batch_size,
            shuffle=False, num_workers=num_workers
        )
    else:
        if train_dir and os.path.exists(train_dir):
            train_dataset = ImageFolderDataset(
                train_dir, transform=get_train_transforms(image_size)
            )
            loaders["train"] = DataLoader(
                train_dataset, batch_size=batch_size,
                shuffle=True, num_workers=num_workers
            )
        
        if val_dir and os.path.exists(val_dir):
            val_dataset = ImageFolderDataset(
                val_dir, transform=get_val_transforms(image_size)
            )
            loaders["val"] = DataLoader(
                val_dataset, batch_size=batch_size,
                shuffle=False, num_workers=num_workers
            )
        
        if test_dir and os.path.exists(test_dir):
            test_dataset = ImageFolderDataset(
                test_dir, transform=get_test_transforms(image_size)
            )
            loaders["test"] = DataLoader(
                test_dataset, batch_size=batch_size,
                shuffle=False, num_workers=num_workers
            )
    
    return loaders
