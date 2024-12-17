# U-Net Implementation for Semantic Segmentation

This project implements and extends the U-Net architecture for semantic segmentation, based on the paper ["U-Net: Convolutional Networks for Biomedical Image Segmentation"](https://arxiv.org/abs/1505.04597) by Ronneberger et al. We provide two implementations with different architectural choices and improvements.

## Original Paper Overview

The original U-Net paper introduced a novel architecture for biomedical image segmentation that consisted of:
- A contracting path (encoder) to capture context
- An expanding path (decoder) for precise localization
- Skip connections between corresponding encoder and decoder layers
- No fully connected layers, allowing for variable input sizes
- Data augmentation with elastic deformations

## Our Implementations

### Model V1 (Base U-Net)

Our first implementation closely follows the original architecture with some modifications:
- Input size: 256x256 (vs 572x572 in original paper)
- Padded convolutions to maintain spatial dimensions
- Basic data augmentation (flips, rotations)
- Standard cross-entropy loss
- Adam optimizer instead of SGD with momentum

Performance metrics:
- Average Dice score: 0.249
- Best performing classes:
  - Sky (0.954)
  - Road (0.928)
  - Building (0.768)
  - Sidewalk (0.765)
- Training parameters:
  - Learning rate: 3e-4
  - Batch size: 16
  - Image size: 256x256

### Model V2 (Enhanced U-Net)

Our second implementation incorporates modern deep learning techniques:

Architectural Improvements:
- Residual blocks in encoder and decoder
- Batch normalization after each convolution
- Dropout (10%) for regularization
- Increased feature channels (64 to 1024)
- Larger input size (384x384)

Training Improvements:
- Combined loss function (Cross Entropy + Dice Loss)
- Class weights based on frequency
- Advanced data augmentation:
  - Random resized crops
  - Color jittering
  - Elastic deformations
  - Grid and optical distortions
- AdamW optimizer with weight decay
- Learning rate scheduling with ReduceLROnPlateau

Training parameters:
- Initial learning rate: 3e-4
- Batch size: 8 (reduced due to larger model)
- Weight decay: 0.01
- Image size: 384x384

## Differences from Original Paper

1. Architecture:
   - Original: Basic convolution blocks
   - V1: Similar to original with padded convolutions
   - V2: Added residual connections, batch norm, and dropout

2. Loss Function:
   - Original: Weighted cross-entropy
   - V1: Standard cross-entropy
   - V2: Combined weighted cross-entropy and Dice loss

3. Training Strategy:
   - Original: SGD with momentum=0.99
   - V1: Adam optimizer
   - V2: AdamW with learning rate scheduling

4. Data Augmentation:
   - Original: Heavy focus on elastic deformations
   - V1: Basic geometric transformations
   - V2: Comprehensive augmentation pipeline

## Dataset

We use the CamVid dataset instead of the original paper's biomedical images. CamVid is a road scene understanding dataset with:
- 32 semantic classes
- 367 training images
- 101 validation images
- 233 test images

## Results

### Quantitative Results
- Model V1 average Dice score: 0.249
- Model V2 average Dice score: 0.214

### Qualitative Analysis
Both models show strengths in segmenting:
1. Large, well-defined objects (sky, road, buildings)
2. High-contrast boundaries
3. Common classes with many training examples

Challenges remain in:
1. Small objects (traffic signs, pedestrians)
2. Rare classes
3. Objects with ambiguous boundaries

## Usage

1. Setup environment:
```bash
pip install -r requirements.txt
```

2. Train models:
```bash
# Train V1 model
python src/train.py

# Train V2 model
python src/train_v2.py
```

3. Evaluate models:
```bash
# Evaluate V1 model
python src/evaluate.py

# Evaluate V2 model
python src/evaluate_v2.py
```

4. Visualize feature distributions:
```bash
# V1 features
python src/visualize_features.py

# V2 features
python src/visualize_features_v2.py
```

## Requirements

See `requirements.txt` for complete list of dependencies.

## Future Work

1. Experiment with transformer-based architectures
2. Implement test-time augmentation
3. Try different backbone networks
4. Explore multi-task learning approaches
5. Investigate self-supervised pretraining 