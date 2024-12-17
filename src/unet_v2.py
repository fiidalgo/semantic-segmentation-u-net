import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
        
    def forward(self, x):
        return self.double_conv(x)

class UNetV2(nn.Module):
    def __init__(self, in_channels=3, out_channels=32, features_start=64):
        super().__init__()
        
        self.encoder1 = DoubleConv(in_channels, features_start)
        self.encoder2 = DoubleConv(features_start, features_start * 2)
        self.encoder3 = DoubleConv(features_start * 2, features_start * 4)
        self.encoder4 = DoubleConv(features_start * 4, features_start * 8)
        
        self.bottleneck = DoubleConv(features_start * 8, features_start * 16)
        
        self.up_conv4 = nn.ConvTranspose2d(features_start * 16, features_start * 8, kernel_size=2, stride=2)
        self.decoder4 = DoubleConv(features_start * 16, features_start * 8)
        
        self.up_conv3 = nn.ConvTranspose2d(features_start * 8, features_start * 4, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(features_start * 8, features_start * 4)
        
        self.up_conv2 = nn.ConvTranspose2d(features_start * 4, features_start * 2, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(features_start * 4, features_start * 2)
        
        self.up_conv1 = nn.ConvTranspose2d(features_start * 2, features_start, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(features_start * 2, features_start)
        
        self.final_conv = nn.Conv2d(features_start, out_channels, kernel_size=1)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))
        
        # Decoder
        dec4 = self.up_conv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        
        dec3 = self.up_conv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.up_conv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.up_conv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        return self.final_conv(dec1)

class CombinedLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        # Class weights calculated from dataset statistics
        weights = torch.FloatTensor([
            2.0,  # Sky
            2.0,  # Building
            3.0,  # Column-Pole
            3.0,  # Road
            4.0,  # Sidewalk
            4.0,  # Tree
            5.0,  # Sign-Symbol
            5.0,  # Fence
            6.0,  # Car
            6.0,  # Pedestrian
            7.0,  # Bicyclist
            1.0   # Void
        ]).to(device)
        self.ce_loss = nn.CrossEntropyLoss(weight=weights)
        
    def forward(self, pred, target):
        # Cross Entropy Loss
        ce_loss = self.ce_loss(pred, target)
        
        # Dice Loss
        pred_softmax = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2)
        intersection = (pred_softmax * target_one_hot).sum(dim=(2, 3))
        union = pred_softmax.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        dice_loss = 1 - (2. * intersection + 1e-5) / (union + 1e-5)
        dice_loss = dice_loss.mean()
        
        # Combine losses (0.5 * CE + 0.5 * Dice)
        return 0.5 * ce_loss + 0.5 * dice_loss

def get_loss_function(device):
    """
    Returns a combination of Cross Entropy and Dice Loss with class weights
    """
    return lambda: CombinedLoss(device)