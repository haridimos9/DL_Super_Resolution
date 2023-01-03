import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models import vgg19
import math

padding='same'
padding2 = 'same'

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.feature_extractor = nn.Sequential(
            *list(vgg19_model.features.children())[:18])

    def forward(self, img):
        return self.feature_extractor(img)


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv3d(in_features, in_features,
                      kernel_size=3, stride=1, padding=padding),
            nn.BatchNorm3d(in_features, 0.8),
            nn.PReLU(),
            nn.Conv3d(in_features, in_features,
                      kernel_size=3, stride=1, padding=padding),
            nn.BatchNorm3d(in_features, 0.8),
        )

    def forward(self, x):
        return x + self.conv_block(x)


class GeneratorResNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_residual_blocks=16):
        super(GeneratorResNet, self).__init__()
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # First layer
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=9, stride=1, padding=padding2), nn.PReLU())

        # Residual blocks
        res_blocks = []
        for _ in range(n_residual_blocks):
            res_blocks.append(ResidualBlock(64))
        self.res_blocks = nn.Sequential(*res_blocks)

        # Second conv layer post residual blocks
        self.conv2 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=padding), nn.BatchNorm3d(64, 0.8))

        # Upsampling layers
        upsampling = []
        for out_features in range(2):
            upsampling += [
                # nn.Upsample(scale_factor=2),
                nn.Conv3d(64, 256, 3, 1, padding),
                nn.BatchNorm3d(256),
                # nn.PixelShuffle(upscale_factor=2),
                # nn.PReLU(),
            ]

        self.upsampling = nn.Sequential(
            upsampling[0],

        )

        # Final output layer
        self.conv3 = nn.Sequential(
            nn.Conv3d(64, out_channels, kernel_size=9, stride=1, padding=padding2), nn.Tanh())

    def forward(self, x):
        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        
         
        out = nn.Conv3d(64, 256, 3, 1, padding).to(self.device)(out.to(self.device))        
        
        out = nn.BatchNorm3d(256).to(self.device)(out)
        
        out = self.pixel_shuffle_3d(out)
        
        out = nn.PReLU().to(self.device)(out)
                
        out = nn.Conv3d(64, 256, 3, 1, padding).to(self.device)(out)
        out = nn.BatchNorm3d(256).to(self.device)(out)
        out = self.pixel_shuffle_3d(out)
        out = nn.PReLU().to(self.device)(out)

        out = self.conv3(out)
        return out[:,:,-1,:,:] # Only return the most recent image

    def pixel_shuffle_3d(self, tensor_5_dim, upscale_factor=2):
        '''
        tensor of images with dimensions [N,D,C,H,W]
        N: batch size
        D: time dimensions. Number of stacked images
        C: colour channel
        H, W: height, width
        '''
        upsampled_images_tensor = []
        for img in range(tensor_5_dim.shape[2]):
            current_image = tensor_5_dim[:, :, img, :, :]
            upsampled_images_tensor.append(
                nn.PixelShuffle(upscale_factor=upscale_factor)(current_image))
        upsampled_images_tensor = torch.stack(upsampled_images_tensor)
        return upsampled_images_tensor.permute(1, 2, 0, 3, 4).to(self.device)


class Discriminator(nn.Module):
    def __init__(self, hr_shape, in_channels=3):
        super(Discriminator, self).__init__()

        self.input_shape = hr_shape
        in_height, in_width = self.input_shape
        patch_h, patch_w = int(in_height / 2 ** 4), int(in_width / 2 ** 4)
        self.output_shape = (1, patch_h, patch_w)

        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(nn.Conv2d(in_filters, out_filters,
                          kernel_size=3, stride=1, padding=padding))
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Conv2d(out_filters, out_filters,
                          kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = in_channels
        for i, out_filters in enumerate([64, 128, 256, 512]):
            layers.extend(discriminator_block(
                in_filters, out_filters, first_block=(i == 0)))
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 1,
                      kernel_size=3, stride=1, padding=padding))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)
