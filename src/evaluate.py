import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

from unet import UNet
from dataset import SegmentationDataset, get_val_transform, CAMVID_CLASSES, CAMVID_COLORS

def evaluate_model(model, test_loader, device, save_dir):
    """Evaluate model on test set and save visualizations"""
    model.eval()
    class_dice_scores = {cls: [] for cls in CAMVID_CLASSES}
    
    os.makedirs(os.path.join(save_dir, 'predictions'), exist_ok=True)
    
    with torch.no_grad():
        for i, (images, masks) in enumerate(tqdm(test_loader, desc='Evaluating')):
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            predictions = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(predictions, dim=1)
            
            # Calculate per-class Dice scores
            for class_idx, class_name in enumerate(CAMVID_CLASSES):
                pred_mask = (predictions == class_idx)
                target_mask = (masks == class_idx)
                
                intersection = (pred_mask & target_mask).sum().float()
                union = pred_mask.sum() + target_mask.sum()
                
                if union > 0:
                    dice = (2. * intersection) / (union + 1e-8)
                    class_dice_scores[class_name].append(dice.item())
            
            # Save visualizations for first batch
            if i == 0:
                save_predictions(images, masks, predictions, save_dir)
    
    # Calculate and print average Dice score for each class
    print("\nPer-class Dice Scores:")
    total_dice = 0
    num_classes = 0
    for class_name, scores in class_dice_scores.items():
        if scores:  # Only consider classes that appeared in the test set
            avg_dice = np.mean(scores)
            print(f"{class_name}: {avg_dice:.4f}")
            total_dice += avg_dice
            num_classes += 1
    
    avg_dice = total_dice / num_classes if num_classes > 0 else 0
    print(f"\nOverall Average Dice Score: {avg_dice:.4f}")
    
    return avg_dice, class_dice_scores

def save_predictions(images, masks, predictions, save_dir, num_samples=5):
    """Save visualization of predictions"""
    images = images.cpu()
    masks = masks.cpu()
    predictions = predictions.cpu()
    
    # Denormalize images
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    images = images * std + mean
    
    def create_color_mask(indices):
        """Convert class indices to RGB mask"""
        color_mask = np.zeros((*indices.shape, 3), dtype=np.uint8)
        for class_idx, color in enumerate(CAMVID_COLORS):
            mask = (indices == class_idx)
            color_mask[mask] = color
        return color_mask
    
    for idx in range(min(num_samples, len(images))):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot original image
        img = images[idx].permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        axes[0].imshow(img)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Plot ground truth mask
        mask_rgb = create_color_mask(masks[idx].numpy())
        axes[1].imshow(mask_rgb)
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        # Plot prediction
        pred_rgb = create_color_mask(predictions[idx].numpy())
        axes[2].imshow(pred_rgb)
        axes[2].set_title('Prediction')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'predictions', f'sample_{idx}.png'))
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Evaluate U-Net model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to saved model')
    parser.add_argument('--test_dir', type=str, default='data/test', help='Path to test data directory')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--image_size', type=int, default=256, help='Image size')
    args = parser.parse_args()
    
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
    
    # Load model
    model = UNet(in_channels=3, out_channels=len(CAMVID_CLASSES)).to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'Loaded model from epoch {checkpoint["epoch"]} with validation loss {checkpoint["val_loss"]:.4f}')
    
    # Create test dataset and loader
    test_dataset = SegmentationDataset(
        image_dir=os.path.join(args.test_dir, 'images'),
        mask_dir=os.path.join(args.test_dir, 'masks'),
        transform=get_val_transform(args.image_size)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Create save directory
    save_dir = os.path.dirname(args.model_path)
    
    # Evaluate model
    avg_dice, class_dice_scores = evaluate_model(model, test_loader, device, save_dir)
    
    # Save results
    results = {
        'avg_dice': avg_dice,
        'class_dice_scores': {k: np.mean(v) if v else 0 for k, v in class_dice_scores.items()},
        'model_path': args.model_path,
        'test_dir': args.test_dir,
        'image_size': args.image_size,
        'batch_size': args.batch_size
    }
    
    with open(os.path.join(save_dir, 'evaluation_results.txt'), 'w') as f:
        for key, value in results.items():
            if key == 'class_dice_scores':
                f.write(f'{key}:\n')
                for class_name, score in value.items():
                    f.write(f'  {class_name}: {score:.4f}\n')
            else:
                f.write(f'{key}: {value}\n')

if __name__ == '__main__':
    main() 