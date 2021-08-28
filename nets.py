import torch
import torch.nn as nn
from torch.nn.functional import relu
from utils import load_data
from torchvision.models import resnet50
import math
import torch.nn.functional as F

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(), 
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        channel_att_raw = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                # print("avg pool: ", avg_pool)
                channel_att_raw = self.mlp( avg_pool )
                # print("Channel att raw : ", channel_att_raw)
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            else:
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
                # print(channel_att_sum)
            else:
                channel_att_sum = channel_att_sum + channel_att_raw
        # print(f"dimension obtained before unsqueezing: {channel_att_sum}")
        scale = torch.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        # print(scale.shape)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out

class SE(nn.Module):
    "credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4"
    def __init__(self, c, r=16):
        super(SE, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )
        
        return

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)

        return x * y.expand_as(x)


class Coder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Coder, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)

        return

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = relu(x, inplace=True)
        x = self.conv2(x)
        x = self.bn2(x)
        return relu(x, inplace=True)


class SE_UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32, r=16):
        super(SE_UNet, self).__init__()

        features = init_features
        self.encoder1 = Coder(in_channels, features)
        self.se1 = SE(features * 2, r)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = Coder(features, features * 2)
        self.se2 = SE(features * 4, r)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = Coder(features * 2, features * 4)
        self.se3 = SE(features * 8, r)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = Coder(features * 4, features * 8)
        self.se4 = SE(features * 16, r)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = Coder(features * 8, features * 16)

        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = Coder((features * 8) * 2, features * 8)
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = Coder((features * 4) * 2, features * 4)
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = Coder((features * 2) * 2, features * 2)
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = Coder(features * 2, features)

        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)
        
        self.training_losses = []
        self.eval_losses = []
        
        return

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.se4(dec4)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.se3(dec3)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.se2(dec2)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.se1(dec1)
        dec1 = self.decoder1(dec1)

        return torch.sigmoid(self.conv(dec1))
    

'''
The output size of each block in ResNet50
input: torch.Size([16, 3, 256, 256])
    0 conv:  torch.Size([16, 64, 128, 128])
    1 batch: torch.Size([16, 64, 128, 128])
    2 relu:  torch.Size([16, 64, 128, 128])
    3 pool:  torch.Size([16, 64, 64, 64])
    4 block: torch.Size([16, 256, 64, 64])
    5 block: torch.Size([16, 512, 32, 32])
    6 block: torch.Size([16, 1024, 16, 16])
    7 block: torch.Size([16, 2048, 8, 8])
'''

class Res_SE_UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, r=16):
        super(Res_SE_UNet, self).__init__()
        
        # Get the pretrained ResNet50 and list the sub blocks
        resnet = resnet50(pretrained=True)
        children = list(resnet.children())

        self.encoder1 = nn.Identity() # Out: (batchsize, 3, 256, 256)
        self.se1 = SE(3 + 64, r) # 3 Channels at this layer
        self.encoder2 = nn.Sequential(*children[:3]) # Out: (batchsize, 64, 128, 128)
        self.se2 = SE(64 * 2, r)
        self.encoder3 = nn.Sequential(*children[3:5]) # Out: (batchsize, 256, 64, 64)
        self.se3 = SE(256 * 2, r)
        self.encoder4 = children[5] # Out: (batchsize, 512, 32, 32)
        self.se4 = SE(512 * 2, r)
        self.encoder5 = children[6] # Out: (batchsize, 1024, 16, 16)
        self.se5 = SE(1024 * 2, r)
        
        self.bottleneck = children[7] # Out: (batchsize, 2048, 8, 8)

        self.upconv5 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2) # Out: (batchsize, 1024, 16, 16)
        self.decoder5 = Coder(1024 * 2, 1024) # Out: (batchsize, 1024, 16, 16)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2) # Out: (batchsize, 512, 32, 32)
        self.decoder4 = Coder(512 * 2, 512) # Out: (batchsize, 512, 32, 32)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2) # Out: (batchsize, 256, 64, 64)
        self.decoder3 = Coder(256 * 2, 256) # Out: (batchsize, 256, 64, 64)
        self.upconv2 = nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2) # Out: (batchsize, 64, 128, 128)
        self.decoder2 = Coder(64 * 2, 64) # Out: (batchsize, 64, 128, 128)
        self.upconv1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2) # Out: (batchsize, 64, 256, 256)
        self.decoder1 = Coder(64 + 3, 64) # Out: (batchsize, 64, 128, 128)
        
        self.conv = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1) # Out: (batchsize, 1, 256, 256)
        
        self.training_losses = []
        self.eval_losses = []
        self.training_iou = []
        self.eval_iou = []
        
        return

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        enc5 = self.encoder5(enc4)
        
        bottleneck = self.bottleneck(enc5)

#         enc1 = self.se1(enc1)
#         enc2 = self.se2(enc2)
#         enc3 = self.se3(enc3)
#         enc4 = self.se4(enc4)
#         enc5 = self.se5(enc5)
        
        dec5 = self.upconv5(bottleneck)
        dec5 = torch.cat((dec5, enc5), dim=1)
        dec5 = self.se5(dec5)
        dec5 = self.decoder5(dec5)
        dec4 = self.upconv4(dec5)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.se4(dec4)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.se3(dec3)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.se2(dec2)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.se1(dec1)
        dec1 = self.decoder1(dec1)

        return torch.sigmoid(self.conv(dec1))
    
    def encoder_parameters(self):
        '''
        Return the encoder, ResNet,'s parameters
        '''
        encoder_list = nn.ModuleList([
            self.encoder1,
            self.encoder2,
            self.encoder3,
            self.encoder4,
            self.encoder5,
            self.bottleneck
        ])
        
        return encoder_list.parameters()
    
    def decoder_parameters(self):
        '''
        Return all decoder's parameters
        '''
        decoder_list = nn.ModuleList([
            self.se1,
            self.se2,
            self.se3,
            self.se4,
            self.se5,
            self.upconv5,
            self.upconv4,
            self.upconv3,
            self.upconv2,
            self.upconv1,
            self.decoder5,
            self.decoder4,
            self.decoder3,
            self.decoder2,
            self.decoder1,
            self.conv
        ])
        
        return decoder_list.parameters()


class Full_SE_UNet(nn.Module):
    '''
    This model is similar to SE_UNet, but SE_UNet only has SE blocks
    at the middle of each encoder, decoder pairs, this model has SE
    blocks in encoder and decoder cascades.
    '''
    def __init__(self, in_channels=3, out_channels=1, init_features=32, r=16):
        super(Full_SE_UNet, self).__init__()

        features = init_features
        self.encoder1 = Coder(in_channels, features)
        self.se1 = SE(features * 2, r)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.se11 = SE(features, r)
        self.encoder2 = Coder(features, features * 2)
        self.se2 = SE((features * 2) * 2, r)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.se22 = SE(features * 2, r)
        self.encoder3 = Coder(features * 2, features * 4)
        self.se3 = SE((features * 4) * 2, r)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.se33 = SE(features * 4, r)
        self.encoder4 = Coder(features * 4, features * 8)
        self.se4 = SE((features * 8) * 2, r)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.se44 = SE(features * 8, r)

        self.bottleneck = Coder(features * 8, features * 16)

        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = Coder((features * 8) * 2, features * 8)
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = Coder((features * 4) * 2, features * 4)
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = Coder((features * 2) * 2, features * 2)
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = Coder(features * 2, features)

        self.se = SE(features, r)
        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)
        
        return

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.se11(self.pool1(enc1)))
        enc3 = self.encoder3(self.se22(self.pool2(enc2)))
        enc4 = self.encoder4(self.se33(self.pool3(enc3)))

        bottleneck = self.bottleneck(self.se44(self.pool4(enc4)))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.se4(dec4)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.se3(dec3)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.se2(dec2)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.se1(dec1)
        dec1 = self.decoder1(dec1)

        out = self.se(dec1)
        out = self.conv(out)
        out = torch.sigmoid(out)
        
        return out
    
class CBAM_UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32, r=16):
        super(CBAM_UNet, self).__init__()

        features = init_features
        self.encoder1 = Coder(in_channels, features)
        self.se1 = CBAM(features, r)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = Coder(features, features * 2)
        self.se2 = CBAM(features * 2, r)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = Coder(features * 2, features * 4)
        self.se3 = CBAM(features * 4, r)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = Coder(features * 4, features * 8)
        self.se4 = CBAM(features * 8, r)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = Coder(features * 8, features * 16)

        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = Coder((features * 8) * 2, features * 8)
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = Coder((features * 4) * 2, features * 4)
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = Coder((features * 2) * 2, features * 2)
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = Coder(features * 2, features)

        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)
        
        self.training_losses = []
        self.eval_losses = []
        
        return

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        
        enc1 = self.se1(enc1)
        enc2 = self.se2(enc2)
        enc3 = self.se3(enc3)
        enc4 = self.se4(enc4)

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return torch.sigmoid(self.conv(dec1))
    

'''
The output size of each block in ResNet50
input: torch.Size([16, 3, 256, 256])
    0 conv:  torch.Size([16, 64, 128, 128])
    1 batch: torch.Size([16, 64, 128, 128])
    2 relu:  torch.Size([16, 64, 128, 128])
    3 pool:  torch.Size([16, 64, 64, 64])
    4 block: torch.Size([16, 256, 64, 64])
    5 block: torch.Size([16, 512, 32, 32])
    6 block: torch.Size([16, 1024, 16, 16])
    7 block: torch.Size([16, 2048, 8, 8])
'''

class Res_CBAM_UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, r=16):
        super(Res_CBAM_UNet, self).__init__()
        
      # Get the pretrained ResNet50 and list the sub blocks
        resnet = resnet50(pretrained=True)
        children = list(resnet.children())

        self.encoder1 = nn.Identity() # Out: (batchsize, 3, 256, 256)
        self.se1 = CBAM(3 + 64, r) # 3 Channels at this layer
        self.encoder2 = nn.Sequential(*children[:3]) # Out: (batchsize, 64, 128, 128)
        self.se2 = CBAM(64 * 2, r)
        self.encoder3 = nn.Sequential(*children[3:5]) # Out: (batchsize, 256, 64, 64)
        self.se3 = CBAM(256 * 2, r)
        self.encoder4 = children[5] # Out: (batchsize, 512, 32, 32)
        self.se4 = CBAM(512 * 2, r)
        self.encoder5 = children[6] # Out: (batchsize, 1024, 16, 16)
        self.se5 = CBAM(1024 * 2, r)
        
        self.bottleneck = children[7] # Out: (batchsize, 2048, 8, 8)

        self.upconv5 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2) # Out: (batchsize, 1024, 16, 16)
        self.decoder5 = Coder(1024 * 2, 1024) # Out: (batchsize, 1024, 16, 16)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2) # Out: (batchsize, 512, 32, 32)
        self.decoder4 = Coder(512 * 2, 512) # Out: (batchsize, 512, 32, 32)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2) # Out: (batchsize, 256, 64, 64)
        self.decoder3 = Coder(256 * 2, 256) # Out: (batchsize, 256, 64, 64)
        self.upconv2 = nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2) # Out: (batchsize, 64, 128, 128)
        self.decoder2 = Coder(64 * 2, 64) # Out: (batchsize, 64, 128, 128)
        self.upconv1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2) # Out: (batchsize, 64, 256, 256)
        self.decoder1 = Coder(64 + 3, 64) # Out: (batchsize, 64, 128, 128)
        
        self.conv = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1) # Out: (batchsize, 1, 256, 256)
        
        self.training_losses = []
        self.eval_losses = []
        self.training_iou = []
        self.eval_iou = []
        
        return
    
    def encoder_parameters(self):
        '''
        Return the encoder, ResNet,'s parameters
        '''
        encoder_list = nn.ModuleList([
            self.encoder1,
            self.encoder2,
            self.encoder3,
            self.encoder4,
            self.encoder5,
            self.bottleneck
        ])
        
        return encoder_list.parameters()
    
    def decoder_parameters(self):
        '''
        Return all decoder's parameters
        '''
        decoder_list = nn.ModuleList([
            self.se1,
            self.se2,
            self.se3,
            self.se4,
            self.se5,
            self.upconv5,
            self.upconv4,
            self.upconv3,
            self.upconv2,
            self.upconv1,
            self.decoder5,
            self.decoder4,
            self.decoder3,
            self.decoder2,
            self.decoder1,
            self.conv
        ])
        
        return decoder_list.parameters()


    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        enc5 = self.encoder5(enc4)
        
        bottleneck = self.bottleneck(enc5)

#         enc1 = self.se1(enc1)
#         enc2 = self.se2(enc2)
#         enc3 = self.se3(enc3)
#         enc4 = self.se4(enc4)
#         enc5 = self.se5(enc5)
        
        dec5 = self.upconv5(bottleneck)
        dec5 = torch.cat((dec5, enc5), dim=1)
        dec5 = self.se5(dec5)
        dec5 = self.decoder5(dec5)
        dec4 = self.upconv4(dec5)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.se4(dec4)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.se3(dec3)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.se2(dec2)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.se1(dec1)
        dec1 = self.decoder1(dec1)

        return torch.sigmoid(self.conv(dec1))
    
class Full_CBAM_UNet(nn.Module):
    '''
    This model is similar to SE_UNet, but SE_UNet only has SE blocks
    at the middle of each encoder, decoder pairs, this model has SE
    blocks in encoder and decoder cascades.
    '''
    def __init__(self, in_channels=3, out_channels=1, init_features=32, r=16):
        super(Full_SE_UNet, self).__init__()

        features = init_features
        self.encoder1 = Coder(in_channels, features)
        self.se1 = CBAM(features * 2, r)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.se11 = CBAM(features, r)
        self.encoder2 = Coder(features, features * 2)
        self.se2 = CBAM((features * 2) * 2, r)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.se22 = CBAM(features * 2, r)
        self.encoder3 = Coder(features * 2, features * 4)
        self.se3 = CBAM((features * 4) * 2, r)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.se33 = CBAM(features * 4, r)
        self.encoder4 = Coder(features * 4, features * 8)
        self.se4 = CBAM((features * 8) * 2, r)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.se44 = CBAM(features * 8, r)

        self.bottleneck = Coder(features * 8, features * 16)

        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = Coder((features * 8) * 2, features * 8)
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = Coder((features * 4) * 2, features * 4)
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = Coder((features * 2) * 2, features * 2)
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = Coder(features * 2, features)

        self.se = CBAM(features, r)
        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)
        
        return

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.se11(self.pool1(enc1)))
        enc3 = self.encoder3(self.se22(self.pool2(enc2)))
        enc4 = self.encoder4(self.se33(self.pool3(enc3)))

        bottleneck = self.bottleneck(self.se44(self.pool4(enc4)))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.se4(dec4)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.se3(dec3)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.se2(dec2)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.se1(dec1)
        dec1 = self.decoder1(dec1)

        out = self.se(dec1)
        out = self.conv(out)
        out = torch.sigmoid(out)
        
        return out
    