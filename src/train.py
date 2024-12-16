import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
plt.style.use('bmh')

from unet import UNet
from dataset import SegmentationDataset, get_train_transform, get_val_transform, CAMVID_CLASSES

class DiceScore:
    def __init__(self, num_classes):
        self.num_classes = num_classes
    
    def __call__(self, pred, target):
        pred = torch.softmax(pred, dim=1)
        pred = torch.argmax(pred, dim=1)
        
        dice_scores = []
        for class_idx in range(self.num_classes):
            pred_mask = (pred == class_idx)
            target_mask = (target == class_idx)
            
            intersection = (pred_mask & target_mask).sum().float()
            union = pred_mask.sum() + target_mask.sum()
            
            if union > 0:
                dice = (2. * intersection) / (union + 1e-8)
                dice_scores.append(dice.item())
            
        return np.mean(dice_scores)

def plot_metrics(train_losses, val_losses, train_dices, val_dices, save_dir):
    """Plot training and validation metrics"""
    epochs = range(1, len(train_losses) + 1)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_title('Loss vs. Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot Dice scores
    ax2.plot(epochs, train_dices, 'b-', label='Training Dice')
    ax2.plot(epochs, val_dices, 'r-', label='Validation Dice')
    ax2.set_title('Dice Score vs. Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Dice Score')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_metrics.png'))
    plt.close()

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, save_dir):
    """Training loop with validation"""
    best_val_loss = float('inf')
    dice_metric = DiceScore(num_classes=len(CAMVID_CLASSES))
    
    # Lists to store metrics for plotting
    train_losses = []
    val_losses = []
    train_dices = []
    val_dices = []
    
    # Create plots directory
    plots_dir = os.path.join(save_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_dice = 0
        
        for images, masks in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training'):
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_dice += dice_metric(outputs, masks)
        
        avg_train_loss = train_loss / len(train_loader)
        avg_train_dice = train_dice / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_dice = 0
        
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc='Validation'):
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                val_dice += dice_metric(outputs, masks)
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)
        
        # Store metrics for plotting
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_dices.append(avg_train_dice)
        val_dices.append(avg_val_dice)
        
        # Plot and save metrics
        plot_metrics(train_losses, val_losses, train_dices, val_dices, plots_dir)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {avg_train_loss:.4f}, Train Dice: {avg_train_dice:.4f}')
        print(f'Val Loss: {avg_val_loss:.4f}, Val Dice: {avg_val_dice:.4f}')
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'train_dice': avg_train_dice,
                'val_dice': avg_val_dice,
                'train_history': {
                    'losses': train_losses,
                    'dices': train_dices
                },
                'val_history': {
                    'losses': val_losses,
                    'dices': val_dices
                }
            }, os.path.join(save_dir, 'best_model.pth'))

def main():
    # Set device - check for MPS (Apple Silicon GPU) first, then CUDA, then fall back to CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # Hyperparameters
    BATCH_SIZE = 8
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-4
    IMAGE_SIZE = 256
    NUM_WORKERS = 8  # Increased from 4
    PREFETCH_FACTOR = 2  # Load 2 batches per worker in advance
    
    # Create save directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join('models', f'unet_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)
    
    # Create datasets
    train_dataset = SegmentationDataset(
        split='train',
        transform=get_train_transform(IMAGE_SIZE)
    )
    
    val_dataset = SegmentationDataset(
        split='val',
        transform=get_val_transform(IMAGE_SIZE)
    )
    
    # Create data loaders with optimized settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=PREFETCH_FACTOR,
        persistent_workers=True  # Keep workers alive between epochs
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=PREFETCH_FACTOR,
        persistent_workers=True
    )
    
    # Initialize model, criterion, and optimizer
    model = UNet(in_channels=3).to(device)  # Output channels automatically set to number of classes
    criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss for multi-class segmentation
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"Training U-Net for {len(CAMVID_CLASSES)} classes:")
    for i, class_name in enumerate(CAMVID_CLASSES):
        print(f"{i}: {class_name}")
    
    # Train model
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=NUM_EPOCHS,
        device=device,
        save_dir=save_dir
    )

if __name__ == '__main__':
    main() 