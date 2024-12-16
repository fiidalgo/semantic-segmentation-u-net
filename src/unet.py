import torch
import torch.nn as nn
from dataset import CAMVID_CLASSES

class DoubleConv(nn.Module):
    """Double convolution block used in U-Net"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=None, features=[64, 128, 256, 512]):
        """
        U-Net implementation
        Args:
            in_channels (int): Number of input channels (default: 3 for RGB)
            out_channels (int): Number of output channels (if None, uses number of CamVid classes)
            features (list): List of feature dimensions for each level
        """
        super().__init__()
        
        if out_channels is None:
            out_channels = len(CAMVID_CLASSES)
            
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder (downsampling) path
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder (upsampling) path
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        # Final convolution
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder path
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]  # Reverse for easier access

        # Decoder path
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)  # Upsample
            skip_connection = skip_connections[idx//2]

            # Handle cases where input dimensions aren't perfectly divisible by 2
            if x.shape != skip_connection.shape:
                x = torch.nn.functional.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)  # Double convolution

        return self.final_conv(x)

def test_unet():
    """Test function to verify U-Net implementation"""
    x = torch.randn((1, 3, 256, 256))  # Example input: (batch_size, channels, height, width)
    model = UNet(in_channels=3)  # 3 channels for RGB, output channels automatically set
    preds = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {preds.shape}")
    print(f"Number of classes: {len(CAMVID_CLASSES)}")
    assert preds.shape == (1, len(CAMVID_CLASSES), 256, 256), \
        f"Expected output shape (1, {len(CAMVID_CLASSES)}, 256, 256) but got {preds.shape}"
    print("U-Net test passed successfully!")

if __name__ == "__main__":
    test_unet() 