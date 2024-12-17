import os
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import yaml

from unet_v2 import UNetV2
from dataset import SegmentationDataset, get_val_transform, CAMVID_CLASSES, CAMVID_COLORS
from PIL import Image

def calculate_dice_score(pred, target, num_classes):
    """Calculate Dice score for each class"""
    dice_scores = []
    pred = F.softmax(pred, dim=1)
    pred = torch.argmax(pred, dim=1)
    
    for class_idx in range(num_classes):
        pred_mask = (pred == class_idx)
        target_mask = (target == class_idx)
        
        intersection = torch.sum(pred_mask & target_mask).item()
        union = torch.sum(pred_mask).item() + torch.sum(target_mask).item()
        
        if union > 0:
            dice = (2.0 * intersection) / union
        else:
            dice = 0.0
        
        dice_scores.append(dice)
    
    return dice_scores

def save_prediction_image(image, mask, pred, save_path):
    """Save a visualization of the prediction"""
    # Convert prediction to color image
    pred = pred.cpu().numpy()
    pred_color = np.zeros((*pred.shape, 3), dtype=np.uint8)
    for idx, color in enumerate(CAMVID_COLORS):
        pred_color[pred == idx] = color
        
    # Convert ground truth to color image
    mask = mask.cpu().numpy()
    mask_color = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for idx, color in enumerate(CAMVID_COLORS):
        mask_color[mask == idx] = color
    
    # Denormalize and convert input image to uint8
    image = image.cpu().numpy().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = ((image * std + mean) * 255).astype(np.uint8)
    
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot original image
    ax1.imshow(image)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Plot ground truth
    ax2.imshow(mask_color)
    ax2.set_title('Ground Truth')
    ax2.axis('off')
    
    # Plot prediction
    ax3.imshow(pred_color)
    ax3.set_title('Prediction')
    ax3.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def evaluate_model(model_path, test_dir='data/test', image_size=384, batch_size=8, num_samples=5):
    """Evaluate model and save results"""
    # Set device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # Load model
    model = UNetV2(in_channels=3, out_channels=len(CAMVID_CLASSES), features_start=64).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create dataset and dataloader
    test_dataset = SegmentationDataset(
        split='test',
        transform=get_val_transform(image_size)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Calculate Dice scores
    all_dice_scores = []
    sample_count = 0
    
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            batch_dice_scores = calculate_dice_score(outputs, masks, len(CAMVID_CLASSES))
            all_dice_scores.append(batch_dice_scores)
            
            # Save sample predictions
            if sample_count < num_samples:
                save_prediction_image(
                    images[0],
                    masks[0],
                    torch.argmax(outputs[0], dim=0),
                    os.path.join(os.path.dirname(model_path), f'sample_{sample_count}.png')
                )
                sample_count += 1
    
    # Calculate average Dice scores
    avg_dice_scores = np.mean(all_dice_scores, axis=0)
    
    # Save evaluation results
    results = {
        'avg_dice': float(np.mean(avg_dice_scores)),
        'class_dice_scores': {
            class_name: f"{score:.4f}"
            for class_name, score in zip(CAMVID_CLASSES, avg_dice_scores)
        },
        'model_path': model_path,
        'test_dir': test_dir,
        'image_size': image_size,
        'batch_size': batch_size
    }
    
    # Save results to file
    results_path = os.path.join(os.path.dirname(model_path), 'evaluation_results.txt')
    with open(results_path, 'w') as f:
        yaml.dump(results, f, default_flow_style=False)
    
    return results

if __name__ == '__main__':
    # Find the most recent model directory
    model_dirs = [d for d in os.listdir('models') if d.startswith('unet_v2_')]
    if not model_dirs:
        print("No UNet V2 models found!")
        exit(1)
    
    latest_model_dir = max(model_dirs)
    model_path = os.path.join('models', latest_model_dir, 'best_model.pth')
    
    # Run evaluation
    results = evaluate_model(model_path)
    print(f"Evaluation results saved to {os.path.join(os.path.dirname(model_path), 'evaluation_results.txt')}") 