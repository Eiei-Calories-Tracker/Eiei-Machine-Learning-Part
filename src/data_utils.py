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

def get_dataloaders(data_dir, batch_size=16):
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    
    train_tf, val_tf = get_transforms()
    
    train_dataset = datasets.ImageFolder(train_dir, train_tf)
    val_dataset = datasets.ImageFolder(val_dir, val_tf)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader

def get_eval_loader(data_dir, batch_size=32, split='test'):
    split_dir = os.path.join(data_dir, split)
    if not os.path.isdir(split_dir):
        split_dir = os.path.join(data_dir, 'val')
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Evaluation split directory not found in {data_dir}: expected '{split}' or 'val'.")

    _, val_tf = get_transforms()
    eval_dataset = datasets.ImageFolder(split_dir, val_tf)
    return DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def _reservoir_sample(items, sample_size):
    if sample_size <= 0 or not items:
        return []
    sample_size = min(sample_size, len(items))
    reservoir = items[:sample_size]
    for index in range(sample_size, len(items)):
        replace_index = random.randint(0, index)
        if replace_index < sample_size:
            reservoir[replace_index] = items[index]
    return reservoir


def prepare_new_version_from_latest_with_reservoir(
    base_data_dir,
    target_version,
    sample_ratio=0.7,
    train_ratio=0.8,
    val_ratio=0.1,
    seed=42,
):
    random.seed(seed)

    latest_version = get_latest_version(base_data_dir)
    if not latest_version:
        raise ValueError("No latest dataset version found for reservoir sampling.")

    latest_version_dir = os.path.join(base_data_dir, latest_version)
    if not os.path.isdir(latest_version_dir):
        raise FileNotFoundError(f"Latest version directory not found: {latest_version_dir}")

    candidates = []
    valid_extensions = ('.png', '.jpg', '.jpeg')
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(latest_version_dir, split)
        if not os.path.isdir(split_dir):
            continue
        for class_name in os.listdir(split_dir):
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for file_name in os.listdir(class_dir):
                if file_name.lower().endswith(valid_extensions):
                    candidates.append((class_name, os.path.join(class_dir, file_name)))

    if not candidates:
        raise ValueError(f"No images found in latest dataset version: {latest_version}")

    target_count = max(1, int(len(candidates) * sample_ratio))
    sampled_items = _reservoir_sample(candidates, target_count)
    random.shuffle(sampled_items)

    new_data_dir = os.path.join(base_data_dir, target_version)
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(new_data_dir, split), exist_ok=True)

    total = len(sampled_items)
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)
    n_test = total - n_train - n_val

    splits = {
        'train': sampled_items[:n_train],
        'val': sampled_items[n_train:n_train + n_val],
        'test': sampled_items[n_train + n_val:n_train + n_val + n_test],
    }

    for split_name, split_items in splits.items():
        for class_name, src_path in split_items:
            class_dir = os.path.join(new_data_dir, split_name, class_name)
            os.makedirs(class_dir, exist_ok=True)
            shutil.copy(src_path, os.path.join(class_dir, os.path.basename(src_path)))

    return {
        'new_data_dir': new_data_dir,
        'source_version': latest_version,
        'target_version': target_version,
        'sampled_count': total,
        'train_count': len(splits['train']),
        'val_count': len(splits['val']),
        'test_count': len(splits['test']),
    }

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
