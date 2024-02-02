'''
For MEMO implementations of CIFAR-ConvNet
Reference:
https://github.com/wangkiw/ICLR23-MEMO/blob/main/convs/conv_cifar.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class Swish(nn.Module): # Swish(x) = x∗σ(x)
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.sigmoid(input)
# for cifar
def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

class ConvNet2(nn.Module):
    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super().__init__()
        self.out_dim = 64
        self.avgpool = nn.AvgPool2d(8)
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, z_dim),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.avgpool(x)
        features = x.view(x.shape[0], -1)
        return {
            "features":features
        }
    
def get_default_convnet_setting():
    net_width, net_depth, net_act, net_norm, net_pooling = 128, 3, 'relu', 'instancenorm', 'avgpooling'
    return net_width, net_depth, net_act, net_norm, net_pooling

import time

def get_convet3(model, channel, num_classes, im_size=(32, 32)):
    torch.random.manual_seed(int(time.time() * 1000) % 100000)
    net_width, net_depth, net_act, net_norm, net_pooling = get_default_convnet_setting() 
    net = ConvNet3(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)

class ConvNet3(nn.Module):
    def __init__(self, channel, num_classes, net_width, net_depth, net_act, net_norm, net_pooling, im_size = (32,32)):
        super(ConvNet3, self).__init__()

        self.features, shape_feat = self._make_layers(channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size)
        self.out_dim = shape_feat[0]*shape_feat[1]*shape_feat[2]
    def forward(self, x):
        x = self.features(x)
        features = x.view(x.size(0), -1)

        return {
            "features":features
        }


    def _get_activation(self, net_act):
        if net_act == 'sigmoid':
            return nn.Sigmoid()
        elif net_act == 'relu':
            return nn.ReLU(inplace=True)
        elif net_act == 'leakyrelu':
            return nn.LeakyReLU(negative_slope=0.01)
        elif net_act == 'swish':
            return Swish()
        else:
            exit('unknown activation function: %s'%net_act)

    def _get_pooling(self, net_pooling):
        if net_pooling == 'maxpooling':
            return nn.MaxPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'avgpooling':
            return nn.AvgPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'none':
            return None
        else:
            exit('unknown net_pooling: %s'%net_pooling)

    def _get_normlayer(self, net_norm, shape_feat):
        # shape_feat = (c*h*w)
        if net_norm == 'batchnorm':
            return nn.BatchNorm2d(shape_feat[0], affine=True)
        elif net_norm == 'layernorm':
            return nn.LayerNorm(shape_feat, elementwise_affine=True)
        elif net_norm == 'instancenorm':
            return nn.GroupNorm(shape_feat[0], shape_feat[0], affine=True)
        elif net_norm == 'groupnorm':
            return nn.GroupNorm(4, shape_feat[0], affine=True)
        elif net_norm == 'none':
            return None
        else:
            exit('unknown net_norm: %s'%net_norm)

    def _make_layers(self, channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size):
        layers = []
        in_channels = channel
        if im_size[0] == 28:
            im_size = (32, 32)
        shape_feat = [in_channels, im_size[0], im_size[1]]
        for d in range(net_depth):
            layers += [nn.Conv2d(in_channels, net_width, kernel_size=3, padding=3 if channel == 1 and d == 0 else 1)]
            shape_feat[0] = net_width
            if net_norm != 'none':
                layers += [self._get_normlayer(net_norm, shape_feat)]
            layers += [self._get_activation(net_act)]
            in_channels = net_width
            if net_pooling != 'none':
                layers += [self._get_pooling(net_pooling)]
                shape_feat[1] //= 2
                shape_feat[2] //= 2

        return nn.Sequential(*layers), shape_feat

        
class GeneralizedConvNet2(nn.Module):
    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
        )

    def forward(self, x):
        base_features = self.encoder(x)
        return base_features
    
class SpecializedConvNet2(nn.Module):
    def __init__(self,hid_dim=64,z_dim=64):
        super().__init__()
        self.feature_dim = 64
        self.avgpool = nn.AvgPool2d(8)
        self.AdaptiveBlock = conv_block(hid_dim,z_dim)
    
    def forward(self,x):
        base_features = self.AdaptiveBlock(x)
        pooled = self.avgpool(base_features)
        features = pooled.view(pooled.size(0),-1)
        return features

def conv2(pretrained,args):
    return ConvNet2()

def conv3(pretrained,args):
    torch.random.manual_seed(int(time.time() * 1000) % 100000)
    net_width, net_depth, net_act, net_norm, net_pooling = get_default_convnet_setting() 
    return ConvNet3(channel=3, num_classes=None, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm='batchnorm', net_pooling=net_pooling, im_size=(32,32))


def get_conv_a2fc():
    basenet = GeneralizedConvNet2()
    adaptivenet = SpecializedConvNet2()
    return basenet,adaptivenet

if __name__ == '__main__':
    a, b = get_conv_a2fc()
    _base = sum(p.numel() for p in a.parameters())
    _adap = sum(p.numel() for p in b.parameters())
    print(f"conv :{_base+_adap}")
    
    conv2 = conv2()
    conv2_sum = sum(p.numel() for p in conv2.parameters())
    print(f"conv2 :{conv2_sum}")