import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from tqdm import tqdm

from unet_v2 import UNetV2
from dataset import SegmentationDataset, get_val_transform, CAMVID_CLASSES

def extract_features(model, dataloader, device):
    """Extract features from the bottleneck layer of U-Net V2"""
    features = []
    labels = []
    
    model.eval()
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Extracting features"):
            images = images.to(device)
            batch_size, _, height, width = images.shape
            
            # Forward through encoder
            x = images
            enc1 = model.encoder1(x)
            enc2 = model.encoder2(model.pool(enc1))
            enc3 = model.encoder3(model.pool(enc2))
            enc4 = model.encoder4(model.pool(enc3))
            
            # Get bottleneck features
            features_batch = model.bottleneck(model.pool(enc4))
            
            # Convert to feature vectors
            features_batch = features_batch.view(batch_size, -1).cpu().numpy()
            features.extend(features_batch)
            
            # Get most common class for each image (excluding void class)
            for mask in masks:
                unique, counts = np.unique(mask.numpy(), return_counts=True)
                # Remove void class (index 30) if present
                if 30 in unique:
                    void_idx = np.where(unique == 30)[0][0]
                    unique = np.delete(unique, void_idx)
                    counts = np.delete(counts, void_idx)
                # Get most common class
                if len(counts) > 0:
                    most_common = unique[np.argmax(counts)]
                else:
                    most_common = 30  # void class if no other class is present
                labels.append(most_common)
    
    return np.array(features), np.array(labels)

def plot_feature_distribution(features, labels, save_path):
    """Create t-SNE visualization of feature distribution"""
    # Select most common classes for visualization
    class_counts = np.bincount(labels)
    top_classes = np.argsort(class_counts)[-4:]  # Get indices of 4 most common classes
    
    # Filter features and labels for top classes
    mask = np.isin(labels, top_classes)
    features = features[mask]
    labels = labels[mask]
    
    # Reduce dimensionality with t-SNE
    print("Performing t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=2, random_state=42, verbose=1)
    embedding = tsne.fit_transform(features)
    
    # Create scatter plot
    plt.figure(figsize=(12, 8))
    
    # Plot points for each class
    for class_idx in top_classes:
        mask = labels == class_idx
        if np.any(mask):  # Only plot if class exists in dataset
            plt.scatter(embedding[mask, 0], 
                       embedding[mask, 1],
                       label=CAMVID_CLASSES[class_idx],
                       alpha=0.6,
                       s=10)
    
    plt.title('Feature Distribution of CamVid Classes (t-SNE)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

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
    
    # Find the most recent model directory
    model_dirs = [d for d in os.listdir('models') if d.startswith('unet_v2_')]
    if not model_dirs:
        print("No UNet V2 models found!")
        exit(1)
    
    latest_model_dir = max(model_dirs)
    model_path = os.path.join('models', latest_model_dir, 'best_model.pth')
    
    # Load model
    model = UNetV2(in_channels=3, out_channels=len(CAMVID_CLASSES), features_start=64).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create dataset and dataloader
    dataset = SegmentationDataset(
        split='train',  # Use training set for visualization
        transform=get_val_transform(384)  # Use same size as training
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Extract features and get labels
    print("Extracting features from the model...")
    features, labels = extract_features(model, dataloader, device)
    
    # Create visualization
    save_path = os.path.join(os.path.dirname(model_path), 'feature_distribution.png')
    plot_feature_distribution(features, labels, save_path)
    print(f"Feature distribution plot saved to {save_path}")

if __name__ == '__main__':
    main() 