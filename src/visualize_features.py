import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from tqdm import tqdm

from unet import UNet
from dataset import SegmentationDataset, get_val_transform, CAMVID_CLASSES

def extract_features(model, dataloader, device):
    """Extract features from the bottleneck layer of U-Net"""
    features = []
    labels = []
    
    model.eval()
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Extracting features"):
            images = images.to(device)
            batch_size, _, height, width = images.shape
            
            # Get features from the bottleneck layer
            x = images
            skip_connections = []
            
            # Forward through encoder
            for down in model.downs:
                x = down(x)
                skip_connections.append(x)
                x = model.pool(x)
            
            # Get bottleneck features
            features_batch = model.bottleneck(x)
            
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
    print("Performing t-SNE dimensionality reduction...")
    # Reduce dimensionality with t-SNE
    tsne = TSNE(n_components=2, random_state=42, verbose=1)
    embedding = tsne.fit_transform(features)
    
    # Create scatter plot
    plt.figure(figsize=(15, 10))
    
    # Plot points for each class
    for i, class_name in enumerate(CAMVID_CLASSES):
        mask = labels == i
        if np.any(mask):  # Only plot if class exists in dataset
            plt.scatter(embedding[mask, 0], 
                       embedding[mask, 1],
                       label=class_name,
                       alpha=0.6,
                       s=10)
    
    plt.title('Feature Distribution of CamVid Classes (t-SNE)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
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
    
    # Load model
    model_dir = 'models/unet_20241215_191402'
    model = UNet(in_channels=3, out_channels=len(CAMVID_CLASSES)).to(device)
    checkpoint = torch.load(os.path.join(model_dir, 'best_model.pth'), 
                          map_location=device,
                          weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create dataset and dataloader
    dataset = SegmentationDataset(
        split='train',  # Use training set for visualization
        transform=get_val_transform(256)
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
    save_path = os.path.join(model_dir, 'feature_distribution.png')
    plot_feature_distribution(features, labels, save_path)
    print(f"Feature distribution plot saved to {save_path}")

if __name__ == '__main__':
    main() 