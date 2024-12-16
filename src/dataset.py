import os
import pandas as pd
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from functools import lru_cache

def load_camvid_classes():
    """Load class information from class_dict.csv"""
    df = pd.read_csv('data/class_dict.csv')
    classes = df['name'].tolist()
    colors = df[['r', 'g', 'b']].values.tolist()
    return classes, colors

# Load CamVid classes and their corresponding colors
CAMVID_CLASSES, CAMVID_COLORS = load_camvid_classes()

class SegmentationDataset(Dataset):
    def __init__(self, split='train', transform=None):
        """
        Custom Dataset for image segmentation
        Args:
            split (str): One of 'train', 'val', or 'test'
            transform: Optional transform to be applied
        """
        self.image_dir = f'data/{split}'
        self.mask_dir = f'data/{split}_labels'
        self.transform = transform
        
        # Cache file lists
        self.images = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.png')])
        self.masks = sorted([f for f in os.listdir(self.mask_dir) if f.endswith('.png')])
        
        # Create and cache color to class mapping
        self.color_to_class = {tuple(color): idx for idx, color in enumerate(CAMVID_COLORS)}
        self.num_classes = len(CAMVID_CLASSES)
        
        # Cache full paths
        self.image_paths = [os.path.join(self.image_dir, img) for img in self.images]
        self.mask_paths = [os.path.join(self.mask_dir, mask) for mask in self.masks]
        
        # Verify matching pairs
        assert len(self.images) == len(self.masks), \
            f"Found {len(self.images)} images but {len(self.masks)} masks"
    
    @lru_cache(maxsize=32)  # Cache recently converted masks
    def convert_mask_to_class_indices(self, mask_path):
        """Convert RGB mask to class indices"""
        mask = np.array(Image.open(mask_path).convert("RGB"))
        height, width, _ = mask.shape
        class_mask = np.zeros((height, width), dtype=np.int64)
        
        # Convert RGB values to class indices
        for color, class_idx in self.color_to_class.items():
            # Create boolean mask for current color
            color_mask = np.all(mask == color, axis=-1)
            class_mask[color_mask] = class_idx
            
        return class_mask
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Use cached paths
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        # Load image and mask
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = self.convert_mask_to_class_indices(mask_path)
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
            # If mask is already a tensor (from ToTensorV2), use it directly
            if not isinstance(mask, torch.Tensor):
                mask = torch.from_numpy(mask)
        else:
            # Convert to tensor if no transform is applied
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()
            mask = torch.from_numpy(mask)
        
        return image, mask.long()

def get_train_transform(image_size=256):
    """Get training transforms with augmentations"""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.5),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=0.5),
        ], p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def get_val_transform(image_size=256):
    """Get validation transforms without augmentations"""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]) 