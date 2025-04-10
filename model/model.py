import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, in_channels=1, base_channels=8):

        super(Encoder, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, base_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(base_channels)
        
        self.conv2 = nn.Conv3d(base_channels, base_channels * 2, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(base_channels * 2)
        
        self.conv3 = nn.Conv3d(base_channels * 2, base_channels * 4, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm3d(base_channels * 4)
    
    def forward(self, x):
        #  (B, 1, 60, 60, 60)
        x = F.relu(self.bn1(self.conv1(x)))   # (B, base_channels, 60, 60, 60)
        x = F.relu(self.bn2(self.conv2(x)))   # (B, base_channels*2, 60, 60, 60)
        x = F.relu(self.bn3(self.conv3(x)))   # (B, base_channels*4, 60, 60, 60)
        return x


class Decoder(nn.Module):
    def __init__(self, out_channels=8, base_channels=8):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv3d(base_channels * 4, base_channels * 2, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(base_channels * 2)
        
        self.conv2 = nn.Conv3d(base_channels * 2, base_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(base_channels)
        
        self.conv3 = nn.Conv3d(base_channels, out_channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))  # (B, base_channels*2, 60, 60, 60)
        x = F.relu(self.bn2(self.conv2(x)))  # (B, base_channels, 60, 60, 60)
        x = self.conv3(x)  #  (B, out_channels, 60, 60, 60)
        return x


class AutoEncoder3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=8, base_channels=8):
        super(AutoEncoder3D, self).__init__()
        self.encoder = Encoder(in_channels=in_channels, base_channels=base_channels)
        self.decoder = Decoder(out_channels=out_channels, base_channels=base_channels)
    
    def forward(self, x):
        # x: (B, 1, 60, 60, 60)
        z = self.encoder(x)       # z (B, base_channels*4, 60, 60, 60)
        x_hat = self.decoder(z)   # x_hat  (B, out_channels=8, 60, 60, 60)
        return x_hat



def dice_loss(pred, target, smooth=1.0):
    pred = torch.softmax(pred, dim=1)
    intersection = (pred * target).sum(dim=(2, 3, 4))
    union = pred.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4))
    dice = (2. * intersection + smooth) / (union + smooth)
    dice_loss_val = 1 - dice  
    return dice_loss_val.mean()


# if __name__ == "__main__":
#     x = torch.randn(2, 1, 60, 60, 60)
#     target_indices = torch.randint(0, 8, (2, 60, 60, 60))
#     target = F.one_hot(target_indices, num_classes=8).permute(0, 4, 1, 2, 3).float()
#     model = AutoEncoder3D(in_channels=1, out_channels=8, base_channels=8)
#     x_hat = model(x)
#     print("x_hat shape: ", x_hat.shape)  #  (2, 8, 60, 60, 60)
#     loss = dice_loss(x_hat, target)
#     print("Dice Loss: ", loss.item())
#     mse_loss = F.mse_loss(x_hat, target)  
#     print("MSE Loss: ", mse_loss.item())
