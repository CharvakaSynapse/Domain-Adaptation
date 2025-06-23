import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np

class Office31Dataset(Dataset):
    def __init__(self, root, domain, transform=None, weak_transform=None):
        self.root = os.path.join(root, domain)
        self.transform = transform
        self.weak_transform = weak_transform
        self.images, self.labels = [], []
        label_map = {label: idx for idx, label in enumerate(sorted(os.listdir(self.root)))
                     if os.path.isdir(os.path.join(self.root, label))}
        for label in sorted(os.listdir(self.root)):
            label_dir = os.path.join(self.root, label)
            if not os.path.isdir(label_dir): continue
            for img_name in os.listdir(label_dir):
                img_path = os.path.join(label_dir, img_name)
                if os.path.isdir(img_path): continue
                if not img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')): continue
                self.images.append(img_path)
                self.labels.append(label_map[label])
    def __len__(self): return len(self.images)
    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        label = self.labels[idx]
        img_strong = self.transform(img) if self.transform else img
        img_weak = self.weak_transform(img) if self.weak_transform else img
        return img_strong, img_weak, label

def get_data_loaders(data_root, source_domain, target_domain, batch_size, num_classes, train_transform, weak_transform):
    source_dataset = Office31Dataset(data_root, source_domain, transform=train_transform, weak_transform=weak_transform)
    label_counts = np.bincount(source_dataset.labels, minlength=num_classes)
    class_weights = 1. / (label_counts + 1e-6)
    sample_weights = class_weights[source_dataset.labels]
    sampler = WeightedRandomSampler(sample_weights, len(source_dataset), replacement=True)
    source_loader = DataLoader(source_dataset, batch_size=batch_size, sampler=sampler,
                              num_workers=2, drop_last=True)
    target_dataset = Office31Dataset(data_root, target_domain, transform=train_transform, weak_transform=weak_transform)
    target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=2, drop_last=True)
    class_weights = torch.tensor(class_weights, dtype=torch.float).cuda()
    return source_loader, target_loader, class_weights