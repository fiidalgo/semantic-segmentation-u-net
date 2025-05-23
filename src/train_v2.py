import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn.functional as F

from unet_v2 import UNetV2, get_loss_function
from dataset import SegmentationDataset, CAMVID_CLASSES

def get_train_transform(size):
    return A.Compose([
        A.RandomResizedCrop(size, size, scale=(0.5, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=0.5),
        ], p=0.3),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
        A.GaussNoise(p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def get_val_transform(size):
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def calculate_dice_score(pred, target):
    """Calculate Dice score"""
    pred = F.softmax(pred, dim=1)
    pred = torch.argmax(pred, dim=1)
    intersection = torch.sum(pred == target)
    union = torch.sum(pred) + torch.sum(target)
    dice = (2.0 * intersection + 1e-5) / (union + 1e-5)
    return dice.item()

def plot_progress(train_losses, val_losses, train_dice, val_dice, save_dir):
    """Plot training progress"""
    plt.figure(figsize=(15, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Dice score plot
    plt.subplot(1, 2, 2)
    plt.plot(train_dice, label='Training Dice')
    plt.plot(val_dice, label='Validation Dice')
    plt.title('Dice Score vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_progress.png'))
    plt.close()

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, save_dir):
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    train_dice_scores = []
    val_dice_scores = []
    current_lr = optimizer.param_groups[0]['lr']
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_dice = 0
        train_bar = tqdm(train_loader, desc=f'Training Epoch {epoch}')
        
        for images, masks in train_bar:
            images = images.to(device)
            masks = masks.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            dice = calculate_dice_score(outputs, masks)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            train_dice += dice
            train_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'dice': f'{dice:.4f}',
                'lr': f'{current_lr:.2e}'
            })
        
        train_loss /= len(train_loader)
        train_dice /= len(train_loader)
        train_losses.append(train_loss)
        train_dice_scores.append(train_dice)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_dice = 0
        val_bar = tqdm(val_loader, desc=f'Validation Epoch {epoch}')
        
        with torch.no_grad():
            for images, masks in val_bar:
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                dice = calculate_dice_score(outputs, masks)
                
                val_loss += loss.item()
                val_dice += dice
                val_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'dice': f'{dice:.4f}'
                })
        
        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        val_losses.append(val_loss)
        val_dice_scores.append(val_dice)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_dice': train_dice,
                'val_dice': val_dice,
            }, os.path.join(save_dir, 'best_model.pth'))
        
        # Plot and save progress
        plot_progress(train_losses, val_losses, train_dice_scores, val_dice_scores, save_dir)
        
        print(f'Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, '
              f'Train Dice = {train_dice:.4f}, Val Dice = {val_dice:.4f}, LR = {current_lr:.2e}')

def main():
    # Set device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # Create model directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f'models/unet_v2_{timestamp}'
    os.makedirs(save_dir, exist_ok=True)
    
    # Model parameters
    IMAGE_SIZE = 384  # Reduced from 512 for faster training
    BATCH_SIZE = 8    # Reduced from 16 to avoid memory issues
    NUM_EPOCHS = 100
    LEARNING_RATE = 3e-4
    
    # Initialize model
    model = UNetV2(in_channels=3, out_channels=len(CAMVID_CLASSES), features_start=64).to(device)
    
    # Create datasets
    train_dataset = SegmentationDataset(
        split='train',
        transform=get_train_transform(IMAGE_SIZE)
    )
    
    val_dataset = SegmentationDataset(
        split='val',
        transform=get_val_transform(IMAGE_SIZE)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Loss function and optimizer
    criterion = get_loss_function(device)()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # Train model
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=NUM_EPOCHS,
        device=device,
        save_dir=save_dir
    )

if __name__ == '__main__':
    main() 