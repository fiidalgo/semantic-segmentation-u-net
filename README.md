# U-Net Implementation for Image Segmentation

This project implements the U-Net architecture from scratch for image segmentation tasks, as described in the paper "U-Net: Convolutional Networks for Biomedical Image Segmentation" by Ronneberger et al. The implementation is done in PyTorch and includes training, evaluation, and visualization capabilities.

## Project Structure

```
.
├── data/
│   ├── train/
│   │   ├── images/
│   │   └── masks/
│   ├── val/
│   │   ├── images/
│   │   └── masks/
│   └── test/
│       ├── images/
│       └── masks/
├── models/
├── src/
│   ├── unet.py
│   ├── dataset.py
│   ├── train.py
│   └── evaluate.py
├── requirements.txt
└── README.md
```

## Setup

1. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Prepare your dataset:
   - Place your training images in `data/train/images/` and corresponding masks in `data/train/masks/`
   - Place your validation images in `data/val/images/` and corresponding masks in `data/val/masks/`
   - Place your test images in `data/test/images/` and corresponding masks in `data/test/masks/`

## Usage

### Training

To train the model:

```bash
python src/train.py
```

The training script will:
- Create a timestamped directory in `models/` to save checkpoints
- Save the best model based on validation loss
- Print training progress and metrics

### Evaluation

To evaluate a trained model:

```bash
python src/evaluate.py --model_path path/to/model.pth --test_dir data/test
```

The evaluation script will:
- Load the specified model
- Run inference on the test set
- Calculate Dice scores
- Save visualizations of predictions
- Generate an evaluation report

## Model Architecture

The implemented U-Net architecture consists of:
- Encoder path with repeated (3x3 convolution + ReLU) + 2x2 max pooling
- Bottleneck layer
- Decoder path with up-convolutions and concatenation with encoder features
- Final 1x1 convolution to map to output classes

Key features:
- Skip connections between encoder and decoder
- Batch normalization for stable training
- Dice loss function for segmentation
- Data augmentation including elastic deformations

## Results

The model is evaluated using:
- Dice coefficient (main metric)
- Visual comparison of predictions vs ground truth
- Training and validation curves

## References

1. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. In Medical Image Computing and Computer-Assisted Intervention (MICCAI).

## License

This project is licensed under the MIT License - see the LICENSE file for details. 