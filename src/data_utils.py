import os
import random
import shutil
from PIL import Image, ImageFile
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform

def get_dataloaders(data_dir, batch_size=64):
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    
    train_tf, val_tf = get_transforms()
    
    train_dataset = datasets.ImageFolder(train_dir, train_tf)
    val_dataset = datasets.ImageFolder(val_dir, val_tf)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader

def prepare_new_version_data(mock_data_dir, base_data_dir, target_version):
    """
    Creates a new version directory (e.g. data/v2) and splits mock data 80/10/10.
    """
    new_data_dir = os.path.join(base_data_dir, target_version)
    os.makedirs(new_data_dir, exist_ok=True)
    
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(new_data_dir, split), exist_ok=True)
        
    categories = [d for d in os.listdir(mock_data_dir) if os.path.isdir(os.path.join(mock_data_dir, d))]
    
    for cat in categories:
        cat_mock_path = os.path.join(mock_data_dir, cat)
        images = [f for f in os.listdir(cat_mock_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(images)
        
        n = len(images)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)
        
        splits = {
            'train': images[:n_train],
            'val': images[n_train:n_train+n_val],
            'test': images[n_train+n_val:]
        }
        
        for split_name, split_images in splits.items():
            split_cat_dir = os.path.join(new_data_dir, split_name, cat)
            os.makedirs(split_cat_dir, exist_ok=True)
            for img in split_images:
                shutil.copy(os.path.join(cat_mock_path, img), os.path.join(split_cat_dir, img))
                
    return new_data_dir

def get_latest_version(base_data_dir):
    versions = [d for d in os.listdir(base_data_dir) if d.startswith('v') and os.path.isdir(os.path.join(base_data_dir, d))]
    if not versions:
        return None
    # Sort by version number
    versions.sort(key=lambda x: int(x[1:]))
    return versions[-1]
