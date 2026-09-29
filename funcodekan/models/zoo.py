"""
funcodekan.models.zoo (imported from models_all.py / models.py)
================================================================
External SparseKAN model zoo. Model code is unchanged; only the import of
the external `kans` package is guarded (see KANS_AVAILABLE below) and the
sys.path manipulation was removed.

Original header:

Consolidated model definitions for the SparseKAN experiments, covering all
architectures for MNIST, CIFAR-10, CIFAR-100, Tiny ImageNet, and ImageNet.

Merged from: kan_models.py, conv_kagn.py, vgg_kan_imagenet.py,
vgg_kan_cifar.py, and the create_model() dispatcher from model.py.
Behavior is unchanged; only imports were deduplicated and the
`util.load_hf_weights_into_model` import was made lazy (needed only by the
HuggingFace transfer-learning helpers).

Usage:
    from models_all import create_model            # args-based dispatcher
    from models_all import KAN_MLP_MNIST, vggkagn  # or any class directly
"""

from functools import partial
from math import prod
from typing import Dict, List, Tuple, Union, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.hub import load_state_dict_from_url

# The zoo depends on the external `kans` package (efficient_kan, fastkan,
# kagn, pykan, KAN convolutions). Install it or place a `kans/` package on
# PYTHONPATH (e.g. vendored at the repository root). When it is absent this
# module still imports, KANS_AVAILABLE is False, and instantiating any zoo
# model raises an informative ImportError.
try:
    from kans.efficient_kan import KANLinear, KAN as EffKAN
    from kans.kan_convolution.KANConvFast import KAN_Convolutional_Layer
    from kans.fastkan import FastKANLayer
    from kans.kagn_kagn_conv import GRAMLayer, KAGN, KAGNConv2DLayer
    from kans.pykan.KANLayer import KANLayer as PyKANLayer
    from kans.regularization import L1
    KAN = EffKAN  # efficient_kan.KAN, as used by the kan_models.py classes
    KANS_AVAILABLE = True
    KANS_IMPORT_ERROR = None
except ImportError as _e:  # pragma: no cover - depends on environment
    KANS_AVAILABLE = False
    KANS_IMPORT_ERROR = _e

    def _kans_stub(name):
        def _missing(*_a, **_k):
            raise ImportError(
                f"'{name}' requires the external 'kans' package, which is not "
                f"installed (original error: {KANS_IMPORT_ERROR}). Vendor your "
                "kans/ package at the repository root or pip-install it, then "
                "re-run."
            )
        return _missing

    KANLinear = _kans_stub("KANLinear")
    EffKAN = KAN = _kans_stub("KAN (efficient_kan)")
    KAN_Convolutional_Layer = _kans_stub("KAN_Convolutional_Layer")
    FastKANLayer = _kans_stub("FastKANLayer")
    GRAMLayer = _kans_stub("GRAMLayer")
    KAGN = _kans_stub("KAGN")
    KAGNConv2DLayer = _kans_stub("KAGNConv2DLayer")
    PyKANLayer = _kans_stub("PyKANLayer")
    L1 = _kans_stub("L1")


###############################################################################
# From kan_models.py — MLP/LeNet/ConvNet/ResNet KANs (MNIST, CIFAR-10/100, Tiny ImageNet)
###############################################################################

# from kans.kan_convolution.KANConv import KAN_Convolutional_Layer
# from kans.efficient_kan import KANLinear, KAN
# from kans.kan_convolution.KANConv import KAN_Convolutional_Layer

###################################################################################################################################################
###################################################################################################################################################
###################################################################################################################################################
###################################################################################################################################################
###################################################################################################################################################
###################################################################################################################################################    

class KAN_MLP_MNIST(nn.Module):
    def __init__(self,grid_size=5, spline_order=3):
        super().__init__()

        self.first_layer = KANLinear(
                                in_features=28*28,
                                out_features=64,
                                grid_size=grid_size,
                                spline_order=spline_order,
                                        )
        
        self.second_layer = KANLinear(
                                in_features=64,
                                out_features=10,
                                grid_size=grid_size,
                                spline_order=spline_order,
                                        )

    def forward(self, x):
        x = self.first_layer(x.view(-1, 28*28)) # x.view(-1, 28*28)
        x = self.second_layer(x)
        return x
    


class KAN_MLP_MNIST_FASTKAN(nn.Module):
    def __init__(self,grid_size=5, spline_order=3):
        super().__init__()

        self.first_layer = FastKANLayer(
                                input_dim=28*28,
                                output_dim=64,
                                        )
        
        self.second_layer = FastKANLayer(
                                input_dim=64,
                                output_dim=10,
                                        )

    def forward(self, x):
        x = self.first_layer(x.view(-1, 28*28)) # x.view(-1, 28*28)
        x = self.second_layer(x)
        return x


class KAN_MLP_MNIST_GRAM(nn.Module):
    def __init__(self,grid_size=5, spline_order=3):
        super().__init__()

        self.first_layer = GRAMLayer(
                                in_channels=28*28,
                                out_channels=64,
                                        )
        
        self.second_layer = GRAMLayer(
                                in_channels=64,
                                out_channels=10,
                                        )

    def forward(self, x):
        x = self.first_layer(x.view(-1, 28*28)) # x.view(-1, 28*28)
        x = self.second_layer(x)
        return x


class KAN_MLP_MNIST_PyKAN(nn.Module):
    def __init__(self,grid_size=5, spline_order=3):
        super().__init__()

        self.first_layer = PyKANLayer(
                                in_dim=28*28,
                                out_dim=64,
                                        )
        
        self.second_layer = PyKANLayer(
                                in_dim=64,
                                out_dim=10,
                                        )

    def forward(self, x):
        x, _, _, _ = self.first_layer(x.view(x.size(0), -1)) # x.view(-1, 28*28)
        x, _, _, _ = self.second_layer(x)
        return x


class KAN_ALL_MNIST(nn.Module):
    def __init__(
            self,
            layer_sizes,
            num_classes: int = 10,
            input_channels: int = 1,
            degree: int = 3,
            degree_out: int = 3,
            groups: int = 1,
            dropout: float = 0.0,
            dropout_linear: float = 0.0,
            l1_penalty: float = 0.0,
            affine: bool = True,
            norm_layer: nn.Module = nn.BatchNorm2d
    ):
        super(KAN_ALL_MNIST, self).__init__()

        self.layers = nn.Sequential(
            KAGNConv2DLayer(input_channels, layer_sizes[0], kernel_size=3, degree=degree, groups=1, padding=1, stride=1,
                            dilation=1, affine=affine, norm_layer=norm_layer),
            L1(KAGNConv2DLayer(layer_sizes[0], layer_sizes[1], kernel_size=3, degree=degree, groups=groups, padding=1,
                               stride=2, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               l1_penalty),
            L1(KAGNConv2DLayer(layer_sizes[1], layer_sizes[2], kernel_size=3, degree=degree, groups=groups, padding=1,
                               stride=2, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               l1_penalty),
            L1(KAGNConv2DLayer(layer_sizes[2], layer_sizes[3], kernel_size=3, degree=degree, groups=groups, padding=1,
                               stride=1, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               l1_penalty),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.output = KAGN([layer_sizes[3], 28*28], dropout=dropout_linear, first_dropout=True,
                               degree=degree_out)
        
        # pykan
        self.first_layer = PyKANLayer(in_dim=28*28, out_dim=64)
        
        # gram
        self.second_layer = GRAMLayer(in_channels=64, out_channels=64)
        
        # fastkan
        self.third_layer = FastKANLayer(input_dim=64, output_dim=64)
        
        # efficient kan
        self.fourth_layer = KANLinear(in_features=64, out_features=10)

    def forward(self, x):
        x = self.layers(x)
        x = torch.flatten(x, 1)
        x = self.output(x)
        x, _, _, _ = self.first_layer(x.view(x.size(0), -1)) # x.view(-1, 28*28)
        x = self.second_layer(x)
        x = self.third_layer(x)
        x = self.fourth_layer(x)
        return x
    




class KANLeNet_MNIST(nn.Module):
    def __init__(self, grid_size=5, spline_order=3):
        super().__init__()
        self.fc1 = KANLinear(28*28, 128, grid_size, spline_order)
        self.fc2 = KANLinear(128, 64, grid_size, spline_order)
        self.fc3 = KANLinear(64, 10, grid_size, spline_order)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)




class KAN_MLP_CIFAR10(nn.Module):
    def __init__(self,grid_size=5, spline_order=3):
        super().__init__()

        self.first_layer = KANLinear(
                                in_features=3*32*32, # 3072
                                out_features=256,
                                grid_size=grid_size,
                                spline_order=spline_order,
                                        )
        
        self.second_layer = KANLinear(
                                in_features=256,
                                out_features=10,
                                grid_size=grid_size,
                                spline_order=spline_order,)

    def forward(self, x):
        x = self.first_layer(x.view(-1, 3072)) # x.view(-1, 3072) | 3*32*32
        x = self.second_layer(x)
        return x



class KAN_MLP_CIFAR100(nn.Module):
    def __init__(self,grid_size=5, spline_order=3):
        super().__init__()
        self.sc = 4
        self.first_layer = KANLinear(
                                in_features=3*32*32, # 3072
                                out_features=64//self.sc, grid_size=grid_size,spline_order=spline_order)
        
        self.second_layer = KANLinear(in_features=64//self.sc, out_features=64//self.sc, grid_size=grid_size, spline_order=spline_order,)
        self.third_layer = KANLinear(in_features=64//self.sc, out_features=100//self.sc, grid_size=grid_size, spline_order=spline_order)

    def forward(self, x):
        x = self.first_layer(x.view(-1, 3072)) # x.view(-1, 3072) | 3*32*32
        x = self.second_layer(x)
        x = self.third_layer(x)
        return x
    



class KAN_MLP_TinyImageNet_SM(nn.Module):
    def __init__(self,grid_size=5, spline_order=3):
        super().__init__()
        self.sc = 16
        self.first_layer = KANLinear(
                                in_features=3*64*64, # 
                                out_features=256//self.sc,
                                grid_size=grid_size,
                                spline_order=spline_order,
                                        )
        
        self.second_layer = KANLinear(
                                in_features=256//self.sc,
                                out_features=64,
                                grid_size=grid_size,
                                spline_order=spline_order,)
        self.third_layer = KANLinear(
                                in_features=64,
                                out_features=200,
                                grid_size=grid_size,
                                spline_order=spline_order,)

    def forward(self, x):
        x = self.first_layer(x.view(-1, 3*64*64)) # x.view(-1, 3072) | 3*32*32
        x = self.second_layer(x)
        x = self.third_layer(x)
        return x
    


class KAN_MLP_TinyImageNet(nn.Module):
    def __init__(self,grid_size=5, spline_order=3):
        super().__init__()

        self.first_layer = KANLinear(
                                in_features=3*64*64, # 
                                out_features=256,
                                grid_size=grid_size,
                                spline_order=spline_order,
                                        )
        
        self.second_layer = KANLinear(
                                in_features=256,
                                out_features=64,
                                grid_size=grid_size,
                                spline_order=spline_order,)
        self.third_layer = KANLinear(
                                in_features=64,
                                out_features=200,
                                grid_size=grid_size,
                                spline_order=spline_order,)

    def forward(self, x):
        x = self.first_layer(x.view(-1, 3*64*64)) # x.view(-1, 3072) | 3*32*32
        x = self.second_layer(x)
        x = self.third_layer(x)
        return x
    







###################################################################################################################################################
###################################################################################################################################################
###################################################################################################################################################
###################################################################################################################################################
###################################################################################################################################################
###################################################################################################################################################


class KANConvNet_MNIST(nn.Module):
    def __init__(self, grid_size=5, spline_order=3):
        super().__init__()
        self.sc = 2
        self.conv1 = KAN_Convolutional_Layer(1, 16//self.sc, (5,5),
                                             grid_size=grid_size, spline_order=spline_order)
        self.conv2 = KAN_Convolutional_Layer(16//self.sc, 32//self.sc, (5,5),
                                             grid_size=grid_size, spline_order=spline_order)

        in_features = (32//self.sc) * 4 * 4
        self.fc1 = KANLinear(in_features, 128, grid_size, spline_order)
        self.fc2 = KANLinear(128, 10, grid_size, spline_order)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)






# ---------------- CIFAR-10 ---------------- #
class KANConvNet_CIFAR10(nn.Module):
    def __init__(self, grid_size=5, spline_order=3):
        super().__init__()
        self.sc = 2 # with 8 accuracy does not cross 50% top-1
        self.conv1 = KAN_Convolutional_Layer(3, 32//self.sc, (3,3),
                                             grid_size=grid_size, spline_order=spline_order)
        self.conv2 = KAN_Convolutional_Layer(32//self.sc, 64//self.sc, (3,3),
                                             grid_size=grid_size, spline_order=spline_order)
        self.conv3 = KAN_Convolutional_Layer(64//self.sc, 128//self.sc, (3,3),
                                             grid_size=grid_size, spline_order=spline_order)

        # lazy init placeholder
        self.fc1 = None
        self.fc2 = KANLinear(256, 10, grid_size, spline_order)

        self.grid_size = grid_size
        self.spline_order = spline_order

    def _init_fc1(self, x):
        """Initialize fc1 dynamically based on flattened feature size."""
        in_features = x.size(1)
        self.fc1 = KANLinear(in_features, 256, self.grid_size, self.spline_order).to(x.device)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)   # 32 -> 16
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)   # 16 -> 8
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)   # 8 -> 4
        x = x.view(x.size(0), -1)   # flatten

        # lazily initialize fc1 with correct shape
        if self.fc1 is None:
            self._init_fc1(x)

        x = F.relu(self.fc1(x))
        return self.fc2(x)



# ---------------- CIFAR-100 ---------------- #
class KANConvNet_CIFAR100(nn.Module):
    def __init__(self, grid_size=5, spline_order=3):
        super().__init__()
        self.sc = 8
        self.conv1 = KAN_Convolutional_Layer(3, 64//self.sc, (3,3), grid_size=grid_size, spline_order=spline_order)
        self.conv2 = KAN_Convolutional_Layer(64//self.sc, 128//self.sc, (3,3), grid_size=grid_size, spline_order=spline_order)
        self.conv3 = KAN_Convolutional_Layer(128//self.sc, 256//self.sc, (3,3), grid_size=grid_size, spline_order=spline_order)
        self.fc1 = KANLinear(256*4*4 //self.sc, 512 //4, grid_size, spline_order)
        self.fc2 = KANLinear(512//4, 100, grid_size, spline_order)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)   # 32 -> 16
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)   # 16 -> 8
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)   # 8 -> 4
        x = x.view(-1, (256*4*4)//self.sc)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# ---------------- TinyImageNet ---------------- # might need to scale by 1/8, 1/16
class KANConvNet_TinyImageNet(nn.Module):
    def __init__(self, grid_size=5, spline_order=3):
        super().__init__()
        self.sc = 16
        self.conv1 = KAN_Convolutional_Layer(3, 64//self.sc, (3,3), grid_size=grid_size, spline_order=spline_order)
        self.conv2 = KAN_Convolutional_Layer(64//self.sc, 128//self.sc, (3,3), grid_size=grid_size, spline_order=spline_order)
        self.conv3 = KAN_Convolutional_Layer(128//self.sc, 256//self.sc, (3,3), grid_size=grid_size, spline_order=spline_order)
        self.conv4 = KAN_Convolutional_Layer(256//self.sc, 512//self.sc, (3,3), grid_size=grid_size, spline_order=spline_order)
        self.fc1 = KANLinear((512*4*4)//self.sc, 1024//self.sc, grid_size, spline_order)
        self.fc2 = KANLinear(1024//self.sc, 200, grid_size, spline_order)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)   # 64 -> 32
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)   # 32 -> 16
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)   # 16 -> 8
        x = F.max_pool2d(F.relu(self.conv4(x)), 2)   # 8 -> 4
        x = x.view(-1, (512*4*4)//self.sc)
        x = F.relu(self.fc1(x))
        return self.fc2(x)




###################################################################################################################################################
###################################################################################################################################################
###################################################################################################################################################
###################################################################################################################################################
###################################################################################################################################################
###################################################################################################################################################
###################################################################################################################################################
###################################################################################################################################################
###################################################################################################################################################
###################################################################################################################################################
###################################################################################################################################################
###################################################################################################################################################
# KAN Basic Block (like ResNet BasicBlock)


class KANBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1,
                 grid_size=5, spline_order=3):
        super().__init__()
        self.conv1 = KAN_Convolutional_Layer(
            in_channels, out_channels, kernel_size=(3,3),
            stride=(stride,stride), padding=(1,1),
            grid_size=grid_size, spline_order=spline_order
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = KAN_Convolutional_Layer(
            out_channels, out_channels, kernel_size=(3,3),
            stride=(1,1), padding=(1,1),
            grid_size=grid_size, spline_order=spline_order
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut for downsampling
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)
    



def _make_layer(block, in_channels, out_channels, num_blocks, stride,
                grid_size, spline_order):
    layers = []
    layers.append(block(in_channels, out_channels, stride,
                        grid_size, spline_order))
    for _ in range(1, num_blocks):
        layers.append(block(out_channels, out_channels, 1,
                            grid_size, spline_order))
    return nn.Sequential(*layers)



### Use num_classes=10 for CIFAR-10, num_classes=100 for CIFAR-100.
class KANResNet_CIFAR10(nn.Module):
    def __init__(self, block=KANBasicBlock, num_blocks=[3,3,3],
                 num_classes=10, grid_size=5, spline_order=3):
        super().__init__()
        self.sc = 2
        self.in_channels = 16
        self.conv1 = KAN_Convolutional_Layer(
            3, 16//self.sc, (3,3), stride=(1,1), padding=(1,1),
            grid_size=grid_size, spline_order=spline_order
        )
        self.bn1 = nn.BatchNorm2d(16//self.sc)

        self.layer1 = _make_layer(block, 16//self.sc, 16//self.sc, num_blocks[0], 1,
                                  grid_size, spline_order)
        self.layer2 = _make_layer(block, 16//self.sc, 32//self.sc, num_blocks[1], 2,
                                  grid_size, spline_order)
        self.layer3 = _make_layer(block, 32//self.sc, 64//self.sc, num_blocks[2], 2,
                                  grid_size, spline_order)

        self.fc = KANLinear(64//self.sc, num_classes, grid_size, spline_order)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        return self.fc(out)


class KANResNet_CIFAR100(nn.Module):
    def __init__(self, block=KANBasicBlock, num_blocks=[3,3,3],
                 num_classes=100, grid_size=5, spline_order=3):
        super().__init__()
        self.sc = 1
        self.in_channels = 16
        self.conv1 = KAN_Convolutional_Layer(
            3, 16//self.sc, (3,3), stride=(1,1), padding=(1,1),
            grid_size=grid_size, spline_order=spline_order
        )
        self.bn1 = nn.BatchNorm2d(16//self.sc)

        self.layer1 = _make_layer(block, 16//self.sc, 16//self.sc, num_blocks[0], 1,
                                  grid_size, spline_order)
        self.layer2 = _make_layer(block, 16//self.sc, 32//self.sc, num_blocks[1], 2,
                                  grid_size, spline_order)
        self.layer3 = _make_layer(block, 32//self.sc, 64//self.sc, num_blocks[2], 2,
                                  grid_size, spline_order)

        self.fc = KANLinear(64//self.sc, num_classes, grid_size, spline_order)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        return self.fc(out)

## KAN-ResNet-18 (TinyImageNet)
class KANResNet_TinyImageNet(nn.Module):
    def __init__(self, block=KANBasicBlock, num_blocks=[2,2,2,2],
                 num_classes=200, grid_size=5, spline_order=3):
        super().__init__()
        self.sc = 16
        self.in_channels = 64
        self.conv1 = KAN_Convolutional_Layer(
            3, 64//self.sc, (7,7), stride=(2,2), padding=(3,3),
            grid_size=grid_size, spline_order=spline_order
        )
        self.bn1 = nn.BatchNorm2d(64//self.sc)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = _make_layer(block, 64//self.sc, 128//self.sc, num_blocks[0], 1,
                                  grid_size, spline_order)
        self.layer2 = _make_layer(block, 128//self.sc, 256//self.sc, num_blocks[1], 2,
                                  grid_size, spline_order)
        self.layer3 = _make_layer(block, 256//self.sc, 512//self.sc, num_blocks[2], 2,
                                  grid_size, spline_order)
        self.layer4 = _make_layer(block, 512//self.sc, 512//self.sc, num_blocks[3], 2,
                                  grid_size, spline_order)
        self.layer5 = _make_layer(block, 512//self.sc, 256//self.sc, num_blocks[3], 2,
                                  grid_size, spline_order)
        self.fc = KANLinear(256//self.sc, 256//self.sc, grid_size, spline_order)
        self.ffc = KANLinear(256//self.sc, num_classes, grid_size, spline_order)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = F.avg_pool2d(out, (1,1))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.ffc(out)
        return out


## KAN-ResNet-18 (TinyImageNet)
class KANResNet_TinyImageNet_SM(nn.Module):
    def __init__(self, block=KANBasicBlock, num_blocks=[2,2,2,2],
                 num_classes=200, grid_size=5, spline_order=3):
        super().__init__()
        self.sc = 4
        self.in_channels = 64
        self.conv1 = KAN_Convolutional_Layer(
            3, 64//self.sc, (7,7), stride=(2,2), padding=(3,3),
            grid_size=grid_size, spline_order=spline_order
        )
        self.bn1 = nn.BatchNorm2d(64//self.sc)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = _make_layer(block, 64//self.sc, 256//self.sc, num_blocks[0], 1,
                                  grid_size, spline_order)
        self.layer2 = _make_layer(block, 256//self.sc, 256//self.sc, num_blocks[1], 2,
                                  grid_size, spline_order)
        self.layer3 = _make_layer(block, 256//self.sc, 128//self.sc, num_blocks[2], 2,
                                  grid_size, spline_order)
    

        self.fc = KANLinear(128//self.sc, num_classes, grid_size, spline_order)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        return self.fc(out)


def count_parameters(model: nn.Module) -> None:

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters:      {total_params:,}")














#############################################################################
#############################################################################
#############################################################################
#############################################################################
#############################################################################
#############################################################################
#############################################################################
#############################################################################
#############################################################################
#############################################################################
#############################################################################
#############################################################################






class KANConvNet_CIFAR10_C(nn.Module):
    def __init__(self, grid_size=5, spline_order=3, sc=4):
        super().__init__()
        self.sc = sc

        self.conv1 = KAN_Convolutional_Layer(
            3, 64 // self.sc, (3, 3), padding=(1, 1),
            grid_size=grid_size, spline_order=spline_order
        )
        self.bn1 = nn.BatchNorm2d(64 // self.sc)

        self.conv2 = KAN_Convolutional_Layer(
            64 // self.sc, 128 // self.sc, (3, 3), padding=(1, 1),
            grid_size=grid_size, spline_order=spline_order
        )
        self.bn2 = nn.BatchNorm2d(128 // self.sc)

        self.conv3 = KAN_Convolutional_Layer(
            128 // self.sc, 256 // self.sc, (3, 3), padding=(1, 1),
            grid_size=grid_size, spline_order=spline_order
        )
        self.bn3 = nn.BatchNorm2d(256 // self.sc)

        self.fc = KANLinear(256 // self.sc, 10, grid_size, spline_order)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)   # 32→16
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)   # 16→8
        x = F.max_pool2d(F.relu(self.bn3(self.conv3(x))), 2)   # 8→4
        x = F.adaptive_avg_pool2d(x, (1, 1))                   # -> 1x1
        x = x.view(x.size(0), -1)                              # [B, 256//sc]
        return self.fc(x)





class KANConvNet_CIFAR100_C(nn.Module):
    def __init__(self, grid_size=5, spline_order=3, sc=1):
        super().__init__()
        self.sc = sc

        self.conv1 = KAN_Convolutional_Layer(
            3, 64 // self.sc, (3, 3), padding=(1, 1),
            grid_size=grid_size, spline_order=spline_order
        )
        self.bn1 = nn.BatchNorm2d(64 // self.sc)

        self.conv2 = KAN_Convolutional_Layer(
            64 // self.sc, 128 // self.sc, (3, 3), padding=(1, 1),
            grid_size=grid_size, spline_order=spline_order
        )
        self.bn2 = nn.BatchNorm2d(128 // self.sc)

        self.conv3 = KAN_Convolutional_Layer(
            128 // self.sc, 256 // self.sc, (3, 3), padding=(1, 1),
            grid_size=grid_size, spline_order=spline_order
        )
        self.bn3 = nn.BatchNorm2d(256 // self.sc)

        self.fc = KANLinear(256 // self.sc, 100, grid_size, spline_order)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)   # 32→16
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)   # 16→8
        x = F.max_pool2d(F.relu(self.bn3(self.conv3(x))), 2)   # 8→4
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        return self.fc(x)


class KANConvNet_TinyImageNet_C(nn.Module):
    def __init__(self, grid_size=5, spline_order=3, sc=1):
        super().__init__()
        self.sc = sc

        self.conv1 = KAN_Convolutional_Layer(
            3, 64 // self.sc, (3, 3), padding=(1, 1),
            grid_size=grid_size, spline_order=spline_order
        )
        self.bn1 = nn.BatchNorm2d(64 // self.sc)

        self.conv2 = KAN_Convolutional_Layer(
            64 // self.sc, 128 // self.sc, (3, 3), padding=(1, 1),
            grid_size=grid_size, spline_order=spline_order
        )
        self.bn2 = nn.BatchNorm2d(128 // self.sc)

        self.conv3 = KAN_Convolutional_Layer(
            128 // self.sc, 256 // self.sc, (3, 3), padding=(1, 1),
            grid_size=grid_size, spline_order=spline_order
        )
        self.bn3 = nn.BatchNorm2d(256 // self.sc)

        self.conv4 = KAN_Convolutional_Layer(
            256 // self.sc, 512 // self.sc, (3, 3), padding=(1, 1),
            grid_size=grid_size, spline_order=spline_order
        )
        self.bn4 = nn.BatchNorm2d(512 // self.sc)

        self.fc = KANLinear(512 // self.sc, 200, grid_size, spline_order)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)   # 64→32
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)   # 32→16
        x = F.max_pool2d(F.relu(self.bn3(self.conv3(x))), 2)   # 16→8
        x = F.max_pool2d(F.relu(self.bn4(self.conv4(x))), 2)   # 8→4
        x = F.adaptive_avg_pool2d(x, (1, 1))                   # -> 1x1
        x = x.view(x.size(0), -1)
        return self.fc(x)






#####
#####
class KAN_MLPBlock(nn.Module):
    def __init__(self, dim, hidden_dim, grid_size=5, spline_order=3, dropout=0.2):
        super().__init__()
        self.fc1 = KANLinear(dim, hidden_dim, grid_size, spline_order)
        self.fc2 = KANLinear(hidden_dim, dim, grid_size, spline_order)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(self.fc2(x))
        return x + residual








# ARCH_LIST = [
#     # MNIST
#     "kan_mlp_mnist_sm", "kan_mlp_mnist","kan_lenet_mnist_sm", "kan_lenet_mnist","kan_convnet_mnist",
#     
#     

#     # CIFAR-10
#     "kan_mlp_cifar10_sm", "kan_mlp_cifar10",
#     "kan_convnet_cifar10", "kan_resnet_cifar10",

#     # CIFAR-100
#     "kan_mlp_cifar100",
#     "kan_convnet_cifar100", "kan_resnet_cifar100",

#     # TinyImageNet
#     "kan_mlp_tinyimagenet_sm", "kan_mlp_tinyimagenet",
#     "kan_convnet_tinyimagenet", "kan_resnet_tinyimagenet", "kan_resnet_tinyimagenet_sm"
# ]

# def main():

#     model = KANResNet_TinyImageNet()
#     count_parameters(model)


#     ### MNIST models
#     # KAN_MLP_MNIST_SM() -> 127,040
#     # KAN_MLP_MNIST() -> 508,160
#     # KANLeNet_MNIST_SM -> 257,600
#     # KANLeNet_MNIST() -> 1,091,840
#     # KANConvNet_MNIST() -> 1: 800,160 2: 374,480 4: 185,640

#     #CIFAR10
#     # KAN_MLP_CIFAR10_SM() -> 493,120
#     # KAN_MLP_CIFAR10() ->  256: 64 dim: 1,972,480
#     # KANConvNet_CIFAR10() -> 4: 1,396,080  8: 696,440
#     # KANResNet_CIFAR10() > 2: 674,944 4: 170,272


#     #CIFAR100
#     # KAN_MLP_CIFAR100() # 1: 2,071,040   2: 1,009,280  4: 498,080
#     # KANConvNet_CIFAR100() # 8: 843,120 16: 471,160
#     # KANResNet_CIFAR100() # 1: 2,745,088 2: 703,744  4: 184,672

#     ## tinyImagenet
#     # KAN_MLP_TinyImageNet_SM() -> 4: 8,033,280 8: 4,080,640 16: 2,104,320
#     # KAN_MLP_TinyImageNet() -> 31,749,120
#     # KANConvNet_TinyImageNet() -> 8: 1,810,800  16: 517,240
#     # KANResNet_TinyImageNet -> 4: 14,747,200, 8: 3,725,728 16:
#     # KANResNet_TinyImageNet_SM() ->2: 12,740,736 4: 3,229,760



# if __name__ == "__main__":
#     main()



# 🔹 General Rules of Thumb

# MNIST: Easy → 10–20 epochs is often enough for convergence, 30–50 for polished runs.

# CIFAR-10: Moderate → 100–200 epochs typical.

# CIFAR-100: Harder → 200–300 epochs standard (smaller models may converge slower).

# TinyImageNet: Much harder → 100–150 epochs with a strong backbone, but you might push to 200 if resources allow.

# 🔹 Suggested Epochs for Your Models
# MNIST

# Small MLP / LeNet / ConvNets (<1M params): 20–30 epochs.

# Larger MNIST variants (~1M params): 40–50 epochs.
# ⚡ Overtraining is possible — MNIST saturates quickly.

# CIFAR-10

# MLPs (<2M params): 80–120 epochs.

# KANConvNet (~0.7–1.4M params): 120–160 epochs.

# KANResNet (~0.17–0.67M params): 160–200 epochs.
# ⚡ Use cosine learning rate decay or step decay at 50% and 75% of epochs.

# CIFAR-100

# MLPs (~0.5–2M params): 150–200 epochs.

# KANConvNet (~0.5–0.8M params): 180–240 epochs.

# KANResNet (~0.18–0.7M params): 200–300 epochs (CIFAR-100 needs longer).

# TinyImageNet

# MLPs (2–31M params): 80–120 epochs (large MLPs overfit fast).

# KANConvNet (~0.5–1.8M params): 100–150 epochs.

# KANResNet (~0.5–7M params): 120–180 epochs.
# TinyImageNet is noisy → use data augmentation (RandAugment, CutMix, MixUp).

# 🔹 Practical Training Schedule

# Optimizer: SGD w/ momentum (0.9) or AdamW.

# LR schedule:

# Start: 0.1 (SGD) or 3e-4 (AdamW).

# Cosine decay to near 0.

# Batch size: 128–256 (CIFAR), 256 (TinyImageNet).

# Regularization: Weight decay 5e-4, label smoothing (esp. CIFAR-100/TinyImageNet).



# “We train all MNIST models for 30–50 epochs, CIFAR-10 models for 100–200 epochs, CIFAR-100 models for 200–300 epochs, and TinyImageNet models for 120–180 epochs. This follows common practice for these datasets (He et al., 2016; Zagoruyko & Komodakis, 2016). Learning rates are decayed using cosine annealing.”

###############################################################################
# From conv_kagn.py — Simple/Eight-layer ConvKAGN (MNIST, CIFAR-10/100, Tiny ImageNet)
###############################################################################

class testConvKAGN(nn.Module):
    def __init__(
            self,
            layer_sizes,
            num_classes: int = 10,
            input_channels: int = 1,
            degree: int = 3,
            degree_out: int = 3,
            groups: int = 1,
            dropout: float = 0.0,
            dropout_linear: float = 0.0,
            l1_penalty: float = 0.0,
            affine: bool = True,
            norm_layer: nn.Module = nn.BatchNorm2d
    ):
        super(testConvKAGN, self).__init__()

        self.layers = nn.Sequential(
            KAGNConv2DLayer(input_channels, layer_sizes[0], kernel_size=3, degree=degree, groups=1, padding=1, stride=1,
                            dilation=1, affine=affine, norm_layer=norm_layer),
            L1(KAGNConv2DLayer(layer_sizes[0], layer_sizes[1], kernel_size=3, degree=degree, groups=groups, padding=1,
                               stride=2, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               l1_penalty),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        if degree_out < 2:
            self.output = nn.Sequential(nn.Dropout(p=dropout_linear), nn.Linear(layer_sizes[1], num_classes))
        else:
            self.output = KAGN([layer_sizes[1], num_classes], dropout=dropout_linear, first_dropout=True,
                               degree=degree_out)

    def forward(self, x):
        x = self.layers(x)
        x = torch.flatten(x, 1)
        x = self.output(x)
        return x

class SimpleConvKAGN(nn.Module):
    def __init__(
            self,
            layer_sizes,
            num_classes: int = 10,
            input_channels: int = 1,
            degree: int = 3,
            degree_out: int = 3,
            groups: int = 1,
            dropout: float = 0.0,
            dropout_linear: float = 0.0,
            l1_penalty: float = 0.0,
            affine: bool = True,
            norm_layer: nn.Module = nn.BatchNorm2d
    ):
        super(SimpleConvKAGN, self).__init__()

        self.layers = nn.Sequential(
            KAGNConv2DLayer(input_channels, layer_sizes[0], kernel_size=3, degree=degree, groups=1, padding=1, stride=1,
                            dilation=1, affine=affine, norm_layer=norm_layer),
            L1(KAGNConv2DLayer(layer_sizes[0], layer_sizes[1], kernel_size=3, degree=degree, groups=groups, padding=1,
                               stride=2, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               l1_penalty),
            L1(KAGNConv2DLayer(layer_sizes[1], layer_sizes[2], kernel_size=3, degree=degree, groups=groups, padding=1,
                               stride=2, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               l1_penalty),
            L1(KAGNConv2DLayer(layer_sizes[2], layer_sizes[3], kernel_size=3, degree=degree, groups=groups, padding=1,
                               stride=1, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               l1_penalty),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        if degree_out < 2:
            self.output = nn.Sequential(nn.Dropout(p=dropout_linear), nn.Linear(layer_sizes[3], num_classes))
        else:
            self.output = KAGN([layer_sizes[3], num_classes], dropout=dropout_linear, first_dropout=True,
                               degree=degree_out)

    def forward(self, x):
        x = self.layers(x)
        x = torch.flatten(x, 1)
        x = self.output(x)
        return x


class EightSimpleConvKAGN(nn.Module):
    def __init__(
            self,
            layer_sizes,
            num_classes: int = 10,
            input_channels: int = 1,
            degree: int = 3,
            degree_out: int = 3,
            groups: int = 1,
            dropout: float = 0.0,
            dropout_linear: float = 0.0,
            l1_penalty: float = 0.0,
            affine: bool = True,
            norm_layer: nn.Module = nn.BatchNorm2d
    ):
        super(EightSimpleConvKAGN, self).__init__()

        self.layers = nn.Sequential(
            KAGNConv2DLayer(input_channels, layer_sizes[0], kernel_size=3, degree=degree, groups=1, padding=1, stride=1,
                            dilation=1, affine=affine, norm_layer=norm_layer),
            L1(KAGNConv2DLayer(layer_sizes[0], layer_sizes[1], kernel_size=3, degree=degree, groups=groups, padding=1,
                               stride=2, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               l1_penalty),
            L1(KAGNConv2DLayer(layer_sizes[1], layer_sizes[2], kernel_size=3, degree=degree, groups=groups, padding=1,
                               stride=2, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               l1_penalty),
            L1(KAGNConv2DLayer(layer_sizes[2], layer_sizes[3], kernel_size=3, degree=degree, groups=groups, padding=1,
                               stride=1, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               l1_penalty),
            L1(KAGNConv2DLayer(layer_sizes[3], layer_sizes[4], kernel_size=3, degree=degree, groups=groups, padding=1,
                               stride=1, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               l1_penalty),
            L1(KAGNConv2DLayer(layer_sizes[4], layer_sizes[5], kernel_size=3, degree=degree, groups=groups, padding=1,
                               stride=2, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               l1_penalty),
            L1(KAGNConv2DLayer(layer_sizes[5], layer_sizes[6], kernel_size=3, degree=degree, groups=groups, padding=1,
                               stride=1, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               l1_penalty),
            L1(KAGNConv2DLayer(layer_sizes[6], layer_sizes[7], kernel_size=3, degree=degree, groups=groups, padding=1,
                               stride=1, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               l1_penalty),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        if degree_out < 2:
            self.output = nn.Sequential(nn.Dropout(p=dropout_linear), nn.Linear(layer_sizes[7], num_classes))
        else:
            self.output = KAGN([layer_sizes[7], num_classes], dropout=dropout_linear, first_dropout=True,
                               degree=degree_out)

    def forward(self, x):
        x = self.layers(x)
        x = torch.flatten(x, 1)
        x = self.output(x)
        return x

###############################################################################
# From vgg_kan_imagenet.py — VGG-KAGN backbone (ImageNet, Tiny ImageNet) + HF transfer helpers
###############################################################################

## adopted from https://github.com/IvanDrokin/torch-conv-kan/blob/main/reports/imagenet1k-vggs.md
### for only imagenet1k dataset
# Based on this https://pytorch.org/vision/main/_modules/torchvision/models/vgg.html#vgg16




# keeping only two variants for experiments
cfgs: Dict[str, List[Union[str, int]]] = {
    "VGG11v2": [16, "M", 32, "M", 64, 64, "M", 128, 128, "M", 128, 128, "M", 128, 128],
    "VGG11v4": [16, "M", 32, "M", 64, 64, "M", 128, 128, "M", 128, 128, "M", 256, 256],
}


def kagn_conv3x3(
    in_planes: int,
    out_planes: int,
    degree: int = 3,
    groups: int = 1,
    stride: int = 1,
    dilation: int = 1,
    dropout: float = 0.0,
    norm_layer=nn.InstanceNorm2d,
    l1_decay: float = 0.0,
    **norm_kwargs,
) -> nn.Module:
    """3x3 KAGN conv with padding=1 by default (dilation-aware)."""
    m = KAGNConv2DLayer(
        in_planes,
        out_planes,
        degree=degree,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        groups=groups,
        dropout=dropout,
        norm_layer=norm_layer,
        **norm_kwargs,
    )
    if l1_decay and l1_decay > 0:
        m = L1(m, l1_decay)
    return m


class VGGKAGN(nn.Module):
    """
    Minimal VGG-like backbone with KAGNConv2DLayer blocks.

    Matches your printed structure:
      - features: ModuleList([... KAGNConv2DLayer/MaxPool ...])
      - avgpool: AdaptiveAvgPool2d(expected_feature_shape)
      - classifier: Dropout -> Linear
    """

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        *,
        vgg_type: str = "VGG11v4",
        width_scale: int = 1,
        expected_feature_shape: Tuple[int, int] = (1, 1),
        degree: int = 3,
        groups: int = 1,
        dropout: float = 0.0,
        dropout_linear: float = 0.25,
        l1_decay: float = 0.0,
        affine: bool = True,
        norm_layer=nn.InstanceNorm2d,
    ) -> None:
        super().__init__()

        assert vgg_type in cfgs, f"Unknown vgg_type={vgg_type}. Available: {list(cfgs.keys())}"

        conv_fun = partial(
            kagn_conv3x3,
            degree=degree,
            groups=groups,
            dropout=dropout,
            l1_decay=l1_decay,
            norm_layer=norm_layer,
            affine=affine,
        )
        # first conv: no dropout (matches your printed model)
        conv_fun_first = partial(
            kagn_conv3x3,
            degree=degree,
            groups=groups,
            dropout=0.0,
            l1_decay=l1_decay,
            norm_layer=norm_layer,
            affine=affine,
        )

        self.features = self._make_layers(
            cfg=cfgs[vgg_type],
            conv_fun=conv_fun,
            conv_fun_first=conv_fun_first,
            num_input_features=input_channels,
            width_scale=width_scale,
        )
        self.avgpool = nn.AdaptiveAvgPool2d(expected_feature_shape)

        # compute last channel count from cfg (ignore "M")
        last_c = input_channels
        for v in cfgs[vgg_type]:
            if v != "M":
                last_c = cast(int, v) * width_scale

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_linear),
            nn.Linear(last_c * prod(expected_feature_shape), num_classes),
        )

    @staticmethod
    def _make_layers(
        cfg: List[Union[str, int]],
        conv_fun,
        conv_fun_first,
        num_input_features: int,
        width_scale: int,
    ) -> nn.ModuleList:
        layers: List[nn.Module] = []
        in_channels = num_input_features

        for i, v in enumerate(cfg):
            if v == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                continue

            v = cast(int, v)
            out_channels = v * width_scale

            if i == 0:
                layers.append(conv_fun_first(in_channels, out_channels))
            else:
                layers.append(conv_fun(in_channels, out_channels))

            in_channels = out_channels

        return nn.ModuleList(layers)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.features:
            x = layer(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def vggkagn(
    input_channels: int,
    num_classes: int,
    *,
    vgg_type: str = "VGG11v4",
    width_scale: int = 1,
    expected_feature_shape: Tuple[int, int] = (1, 1),
    degree: int = 3,
    groups: int = 1,
    dropout: float = 0.0,
    dropout_linear: float = 0.25,
    l1_decay: float = 0.0,
    affine: bool = True,
    norm_layer=nn.InstanceNorm2d,
) -> VGGKAGN:
    return VGGKAGN(
        input_channels,
        num_classes,
        vgg_type=vgg_type,
        width_scale=width_scale,
        expected_feature_shape=expected_feature_shape,
        degree=degree,
        groups=groups,
        dropout=dropout,
        dropout_linear=dropout_linear,
        l1_decay=l1_decay,
        affine=affine,
        norm_layer=norm_layer,
    )






def freeze_all_but_classifier(model: nn.Module) -> nn.Module:
    # freeze all
    for p in model.parameters():
        p.requires_grad = False
    # unfreeze classifier
    if hasattr(model, "classifier"):
        for p in model.classifier.parameters():
            p.requires_grad = True
    return model

def replace_classifier_head_for_num_classes(model: nn.Module, num_classes: int) -> None:
    # VGGKAGN uses: model.classifier = Sequential(Dropout, Linear) :contentReference[oaicite:8]{index=8}
    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
        if len(model.classifier) >= 1 and isinstance(model.classifier[-1], nn.Linear):
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, num_classes)
            return model
    # common fallback (resnets etc.)
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model
    raise RuntimeError("Don't know how to replace classifier head for this model type.")


    

def imagenet_kagn_v2_loaded_for_tinyimagenet(freeze=True):
    from util import load_hf_weights_into_model
    model = vggkagn(3,
                1000,
                groups=1,
                degree=5,
                dropout=0.15,
                l1_decay=0,
                dropout_linear=0.25,
                width_scale=2,
                vgg_type='VGG11v2',
                expected_feature_shape=(1, 1),
                affine=True)
    model, ckpt_name, missing, unexpected, _ = load_hf_weights_into_model(
                model, repo_id="brivangl/vgg_kagn11_v2", device="cpu"
            )
    model = replace_classifier_head_for_num_classes(model, num_classes=200)
    if freeze:
        model = freeze_all_but_classifier(model)
    return model

    
def imagenet_kagn_v4_loaded_for_tinyimagenet(freeze=True):
    from util import load_hf_weights_into_model
    model = vggkagn(
                3,
                1000,
                groups=1,
                degree=5,
                dropout=0.15,
                l1_decay=0,
                dropout_linear=0.25,
                width_scale=2,
                vgg_type='VGG11v4',
                expected_feature_shape=(1, 1),
                affine=True
            )
    model, ckpt_name, missing, unexpected, _ = load_hf_weights_into_model(
                model, repo_id="brivangl/vgg_kagn11_v4", device="cpu"
            )
    model = replace_classifier_head_for_num_classes(model, num_classes=200)
    if freeze:
        model = freeze_all_but_classifier(model)
    return model

###############################################################################
# From vgg_kan_cifar.py — CIFAR-specific VGG-KAGN (3-pool / 4-pool variants)
###############################################################################

## CIFAR-100 specific VGG-KAGN variants
## Adapted from vgg_kan_imagenet.py for 32x32 input images
## Key difference: Fewer pooling layers to avoid 1x1 spatial dimensions with InstanceNorm





# CIFAR configs: designed for 32x32 input
# With 3 MaxPools: 32 -> 16 -> 8 -> 4 (final spatial size 4x4)
# With 4 MaxPools: 32 -> 16 -> 8 -> 4 -> 2 (final spatial size 2x2)
cfgs_cifar: Dict[str, List[Union[str, int]]] = {
    # 3 pools version (safer, ends at 4x4)
    "VGG11v2_cifar": [32, "M", 64, "M", 128, 128, "M", 256, 256, 256, 256, 256, 256],
    "VGG11v4_cifar": [32, "M", 64, "M", 128, 128, "M", 256, 256, 256, 256, 512, 512],
    
    # 4 pools version (ends at 2x2, still safe for InstanceNorm)
    "VGG11v2_cifar_4pool": [32, "M", 64, "M", 128, 128, "M", 256, 256, "M", 256, 256, 256, 256],
    "VGG11v4_cifar_4pool": [32, "M", 64, "M", 128, 128, "M", 256, 256, "M", 256, 256, 512, 512],
}


def kagn_conv3x3_cifar(
    in_planes: int,
    out_planes: int,
    degree: int = 3,
    groups: int = 1,
    stride: int = 1,
    dilation: int = 1,
    dropout: float = 0.0,
    norm_layer=nn.InstanceNorm2d,
    l1_decay: float = 0.0,
    **norm_kwargs,
) -> nn.Module:
    """3x3 KAGN conv with padding=1 by default (dilation-aware)."""
    m = KAGNConv2DLayer(
        in_planes,
        out_planes,
        degree=degree,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        groups=groups,
        dropout=dropout,
        norm_layer=norm_layer,
        **norm_kwargs,
    )
    if l1_decay and l1_decay > 0:
        m = L1(m, l1_decay)
    return m


class VGGKAGNCifar(nn.Module):
    """
    VGG-like backbone with KAGNConv2DLayer blocks, designed for CIFAR (32x32).
    
    Key differences from ImageNet version:
    - Fewer pooling layers to maintain spatial dimensions > 1x1
    - Adjusted channel configurations for smaller input
    - expected_feature_shape accounts for reduced pooling
    """

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        *,
        vgg_type: str = "VGG11v2_cifar",
        width_scale: int = 1,
        expected_feature_shape: Tuple[int, int] = (4, 4),  # 4x4 for 3 pools, 2x2 for 4 pools
        degree: int = 3,
        groups: int = 1,
        dropout: float = 0.0,
        dropout_linear: float = 0.25,
        l1_decay: float = 0.0,
        affine: bool = True,
        norm_layer=nn.InstanceNorm2d,
    ) -> None:
        super().__init__()

        assert vgg_type in cfgs_cifar, f"Unknown vgg_type={vgg_type}. Available: {list(cfgs_cifar.keys())}"

        conv_fun = partial(
            kagn_conv3x3_cifar,
            degree=degree,
            groups=groups,
            dropout=dropout,
            l1_decay=l1_decay,
            norm_layer=norm_layer,
            affine=affine,
        )
        # first conv: no dropout (matches original model)
        conv_fun_first = partial(
            kagn_conv3x3_cifar,
            degree=degree,
            groups=groups,
            dropout=0.0,
            l1_decay=l1_decay,
            norm_layer=norm_layer,
            affine=affine,
        )

        self.features = self._make_layers(
            cfg=cfgs_cifar[vgg_type],
            conv_fun=conv_fun,
            conv_fun_first=conv_fun_first,
            num_input_features=input_channels,
            width_scale=width_scale,
        )
        self.avgpool = nn.AdaptiveAvgPool2d(expected_feature_shape)

        # compute last channel count from cfg (ignore "M")
        last_c = input_channels
        for v in cfgs_cifar[vgg_type]:
            if v != "M":
                last_c = cast(int, v) * width_scale

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_linear),
            nn.Linear(last_c * prod(expected_feature_shape), num_classes),
        )

    @staticmethod
    def _make_layers(
        cfg: List[Union[str, int]],
        conv_fun,
        conv_fun_first,
        num_input_features: int,
        width_scale: int,
    ) -> nn.ModuleList:
        layers: List[nn.Module] = []
        in_channels = num_input_features

        for i, v in enumerate(cfg):
            if v == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                continue

            v = cast(int, v)
            out_channels = v * width_scale

            if i == 0:
                layers.append(conv_fun_first(in_channels, out_channels))
            else:
                layers.append(conv_fun(in_channels, out_channels))

            in_channels = out_channels

        return nn.ModuleList(layers)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.features:
            x = layer(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def vggkagn_cifar(
    input_channels: int,
    num_classes: int,
    *,
    vgg_type: str = "VGG11v2_cifar",
    width_scale: int = 1,
    expected_feature_shape: Tuple[int, int] = (4, 4),
    degree: int = 3,
    groups: int = 1,
    dropout: float = 0.0,
    dropout_linear: float = 0.25,
    l1_decay: float = 0.0,
    affine: bool = True,
    norm_layer=nn.InstanceNorm2d,
) -> VGGKAGNCifar:
    """
    Factory function to create CIFAR-specific VGG-KAGN models.
    
    Args:
        input_channels: Number of input channels (3 for CIFAR)
        num_classes: Number of output classes (100 for CIFAR-100, 10 for CIFAR-10)
        vgg_type: One of 'VGG11v2_cifar', 'VGG11v4_cifar', 'VGG11v2_cifar_4pool', 'VGG11v4_cifar_4pool'
        width_scale: Channel width multiplier
        expected_feature_shape: Spatial size after avgpool (use (4,4) for 3-pool, (2,2) for 4-pool)
        degree: Degree for KAGN layers
        groups: Number of groups for grouped convolution
        dropout: Dropout rate for conv layers
        dropout_linear: Dropout rate for classifier
        l1_decay: L1 regularization decay
        affine: Whether to use affine parameters in normalization
        norm_layer: Normalization layer class
    
    Returns:
        VGGKAGNCifar model instance
    """
    return VGGKAGNCifar(
        input_channels,
        num_classes,
        vgg_type=vgg_type,
        width_scale=width_scale,
        expected_feature_shape=expected_feature_shape,
        degree=degree,
        groups=groups,
        dropout=dropout,
        dropout_linear=dropout_linear,
        l1_decay=l1_decay,
        affine=affine,
        norm_layer=norm_layer,
    )


# Convenience functions for common configurations
def kagn_v2_cifar100(
    degree: int = 5,
    dropout: float = 0.15,
    dropout_linear: float = 0.25,
    width_scale: int = 1,
    use_4_pools: bool = False,
) -> VGGKAGNCifar:
    """
    KAGN v2 model for CIFAR-100.
    
    Args:
        degree: Degree for KAGN layers (default 5 to match ImageNet pretrained)
        dropout: Dropout rate for conv layers
        dropout_linear: Dropout rate for classifier
        width_scale: Channel width multiplier
        use_4_pools: If True, use 4 pooling layers (2x2 final), else 3 pools (4x4 final)
    
    Returns:
        VGGKAGNCifar model for CIFAR-100
    """
    vgg_type = "VGG11v2_cifar_4pool" if use_4_pools else "VGG11v2_cifar"
    feature_shape = (2, 2) if use_4_pools else (4, 4)
    
    return vggkagn_cifar(
        input_channels=3,
        num_classes=100,
        vgg_type=vgg_type,
        width_scale=width_scale,
        expected_feature_shape=feature_shape,
        degree=degree,
        groups=1,
        dropout=dropout,
        dropout_linear=dropout_linear,
        l1_decay=0,
        affine=True,
        norm_layer=nn.InstanceNorm2d,
    )


def kagn_v4_cifar100(
    degree: int = 5,
    dropout: float = 0.15,
    dropout_linear: float = 0.25,
    width_scale: int = 1,
    use_4_pools: bool = False,
) -> VGGKAGNCifar:
    """
    KAGN v4 model for CIFAR-100.
    
    Args:
        degree: Degree for KAGN layers (default 5 to match ImageNet pretrained)
        dropout: Dropout rate for conv layers
        dropout_linear: Dropout rate for classifier
        width_scale: Channel width multiplier
        use_4_pools: If True, use 4 pooling layers (2x2 final), else 3 pools (4x4 final)
    
    Returns:
        VGGKAGNCifar model for CIFAR-100
    """
    vgg_type = "VGG11v4_cifar_4pool" if use_4_pools else "VGG11v4_cifar"
    feature_shape = (2, 2) if use_4_pools else (4, 4)
    
    return vggkagn_cifar(
        input_channels=3,
        num_classes=100,
        vgg_type=vgg_type,
        width_scale=width_scale,
        expected_feature_shape=feature_shape,
        degree=degree,
        groups=1,
        dropout=dropout,
        dropout_linear=dropout_linear,
        l1_decay=0,
        affine=True,
        norm_layer=nn.InstanceNorm2d,
    )


def kagn_v2_cifar10(
    degree: int = 5,
    dropout: float = 0.15,
    dropout_linear: float = 0.25,
    width_scale: int = 1,
    use_4_pools: bool = False,
) -> VGGKAGNCifar:
    """KAGN v2 model for CIFAR-10."""
    vgg_type = "VGG11v2_cifar_4pool" if use_4_pools else "VGG11v2_cifar"
    feature_shape = (2, 2) if use_4_pools else (4, 4)
    
    return vggkagn_cifar(
        input_channels=3,
        num_classes=10,
        vgg_type=vgg_type,
        width_scale=width_scale,
        expected_feature_shape=feature_shape,
        degree=degree,
        groups=1,
        dropout=dropout,
        dropout_linear=dropout_linear,
        l1_decay=0,
        affine=True,
        norm_layer=nn.InstanceNorm2d,
    )


def kagn_v4_cifar10(
    degree: int = 5,
    dropout: float = 0.15,
    dropout_linear: float = 0.25,
    width_scale: int = 1,
    use_4_pools: bool = False,
) -> VGGKAGNCifar:
    """KAGN v4 model for CIFAR-10."""
    vgg_type = "VGG11v4_cifar_4pool" if use_4_pools else "VGG11v4_cifar"
    feature_shape = (2, 2) if use_4_pools else (4, 4)
    
    return vggkagn_cifar(
        input_channels=3,
        num_classes=10,
        vgg_type=vgg_type,
        width_scale=width_scale,
        expected_feature_shape=feature_shape,
        degree=degree,
        groups=1,
        dropout=dropout,
        dropout_linear=dropout_linear,
        l1_decay=0,
        affine=True,
        norm_layer=nn.InstanceNorm2d,
    )


# if __name__ == "__main__":
#     # Test the model
#     print("Testing KAGN v2 CIFAR-100 (3 pools, 4x4 final):")
#     model = kagn_v2_cifar100(use_4_pools=False)
#     print(model)
    
#     # Test forward pass
#     x = torch.randn(2, 3, 32, 32)
#     with torch.no_grad():
#         out = model(x)
#     print(f"Input shape: {x.shape}")
#     print(f"Output shape: {out.shape}")
#     print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
#     print("\n" + "="*60 + "\n")
    
#     print("Testing KAGN v2 CIFAR-100 (4 pools, 2x2 final):")
#     model_4pool = kagn_v2_cifar100(use_4_pools=True)
#     with torch.no_grad():
#         out_4pool = model_4pool(x)
#     print(f"Input shape: {x.shape}")
#     print(f"Output shape: {out_4pool.shape}")
#     print(f"Parameters: {sum(p.numel() for p in model_4pool.parameters()):,}")

###############################################################################
# Paper-benchmark models — best configurations reported in KANELÉ (FPGA '26),
# VIKIN, and PDR-KAN, for Wine, Dry Bean, JSC OpenML, and Traffic (California)
###############################################################################

class _KANStack(nn.Module):
    """Sequential KANLinear stack with per-paper grid/order/range settings.

    NOTE: grid_range follows the standard efficient-kan signature; if your
    kans.efficient_kan fork predates that kwarg, drop it (default [-1,1]).
    """

    def __init__(self, dims, grid_size, spline_order, grid_range):
        super().__init__()
        self.layers = nn.ModuleList([
            KANLinear(in_features=dims[i], out_features=dims[i + 1],
                      grid_size=grid_size, spline_order=spline_order,
                      grid_range=grid_range)
            for i in range(len(dims) - 1)
        ])

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for layer in self.layers:
            x = layer(x)
        return x


class KAN_Wine(_KANStack):
    """KANELÉ Table 2 Wine: [13,4,3], G=6, S=3, domain [-8,8].
    Reported: 98.1% FP, 98.2% quantized+pruned (n_l=[6,7,8], T=0);
    MLP FP baseline 96.3%."""

    def __init__(self):
        super().__init__([13, 4, 3], grid_size=6, spline_order=3,
                         grid_range=[-8, 8])


class KAN_DryBean(_KANStack):
    """KANELÉ Table 2 Dry Bean: [16,2,7], G=6, S=3, domain [-8,8].
    Reported: 92.2% FP, 92.1% quantized (n_l=[6,6,8], T=0); MLP 90.9%."""

    def __init__(self):
        super().__init__([16, 2, 7], grid_size=6, spline_order=3,
                         grid_range=[-8, 8])


class KAN_JSC_OpenML(_KANStack):
    """KANELÉ Table 2 JSC OpenML: [16,8,5], G=40, S=10, domain [-2,2].
    Reported: 76.5% FP, 76.0% quantized+pruned (n_l=[6,7,6], T=0.9).
    Unusually expressive splines (order 10, grid 40) + aggressive pruning."""

    def __init__(self):
        super().__init__([16, 8, 5], grid_size=40, spline_order=10,
                         grid_range=[-2, 2])


class KAN_Traffic(_KANStack):
    """VIKIN / PDR-KAN Table I best KAN: 3-layer [72,32,96], 43k params,
    silu + B-spline, K=3, spline domain [-1,1], FP16 at inference.
    G=4 -> MSE 6.06e-4 (Table I); G=16 -> MSE 4.37e-4 (PDR-KAN Table II,
    3.33x params of G=2 at only 1.16x latency on their accelerator).
    Input is a 72-hour univariate window (single traffic sensor), output the
    next 96 hours: use load_traffic_california(sensors=[i], flatten=True)."""

    def __init__(self, grid_size: int = 16, hidden: bool = True):
        dims = [72, 32, 96] if hidden else [72, 96]  # hidden=False: 2-layer
        super().__init__(dims, grid_size=grid_size, spline_order=3,
                         grid_range=[-1, 1])


class MLP_Traffic_Baseline(nn.Module):
    """VIKIN / PDR-KAN Table I MLP baselines: [72,304,96] (51k, MSE 7.92e-4)
    or [72,304,304,96] (144k, MSE 8.20e-4), ReLU activations."""

    def __init__(self, four_layer: bool = False):
        super().__init__()
        dims = [72, 304, 304, 96] if four_layer else [72, 304, 96]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))


###############################################################################
# From model.py — create_model() dispatcher
###############################################################################

import logging

def create_model(args):
    logger = logging.getLogger()

    model = None
    if args.dataloader.dataset == 'mnist':
        if args.arch == 'kan_mlp_mnist':   ### in experiment
            model = KAN_MLP_MNIST()

        elif args.arch == 'kan_lenet_mnist':
            model = KANLeNet_MNIST()
        elif args.arch == 'kan_convnet_mnist':  ### in experiment
            model = KANConvNet_MNIST()
        elif args.arch == 'kagn_simple_mnist': ### in experiment
            model = SimpleConvKAGN([8 * 4, 16 * 4, 32 * 4, 64 * 4], num_classes=10, input_channels=1,
                          degree=3, groups=4, dropout=0.25, dropout_linear=0.5, l1_penalty=0.00000,
                          degree_out=3) 
       
        elif args.arch == 'kagn_simple_mnist_8_layer': ### was in experiment for arxiv
            model = EightSimpleConvKAGN([8, 16, 32, 64, 128, 128, 256, 256],
                               num_classes=10, input_channels=1,
                               degree=3, groups=1, dropout=0.25, dropout_linear=0.5, l1_penalty=0.00000,
                               degree_out=3)
      
        elif args.arch == 'kan_mlp_mnist_fastkan': # In experiment
            model = KAN_MLP_MNIST_FASTKAN()
        
        elif args.arch == 'kan_mlp_mnist_gram': # In experiment
            model = KAN_MLP_MNIST_GRAM()

        elif args.arch == 'kan_mlp_mnist_pykan': # In experiment
            model = KAN_MLP_MNIST_PyKAN()
        
        elif args.arch == 'kan_test_all_mnist': # for testing, dont use it for actual experiment
            model = KAN_ALL_MNIST([8 * 4, 16 * 4, 32 * 4, 64 * 4], num_classes=10, input_channels=1,
                          degree=3, groups=4, dropout=0.25, dropout_linear=0.5, l1_penalty=0.00000,
                          degree_out=3)

    elif args.dataloader.dataset == 'cifar10':
        if args.arch == 'kan_mlp_cifar10': # in experiment for arxiv
            model = KAN_MLP_CIFAR10()
        elif args.arch == 'kan_convnet_cifar10':
            model = KANConvNet_CIFAR10()
        elif args.arch == 'kan_convnet_cifar10_c': # was in experiment for arxiv
            model = KANConvNet_CIFAR10_C() 
        elif args.arch == 'kan_resnet_cifar10':
            model = KANResNet_CIFAR10(num_classes=10)


        elif args.arch == 'kagn_simple_cifar10':  ## in experiment
            model = SimpleConvKAGN([8 * 4, 16 * 4, 32 * 4, 64 * 4], num_classes=10, input_channels=3,
                          degree=3, groups=4, dropout=0.25, dropout_linear=0.5, l1_penalty=0.00000,
                          degree_out=3)
        
        elif args.arch == 'kagn_simple_cifar10_8_layer':  ## was in experiment for arxiv [8, 16, 32, 64, 128, 128, 256, 256]
            model = EightSimpleConvKAGN([8, 16, 32, 64, 128, 256, 256, 128], # for ICML changed to [8, 16, 32, 64, 128, 256, 256, 128]
                               num_classes=10, input_channels=3,
                               degree=3, groups=1, dropout=0.25, dropout_linear=0.5, l1_penalty=0.00000,
                               degree_out=3)
   

    elif args.dataloader.dataset == 'cifar100':
        if args.arch == 'kan_mlp_cifar100':
            model = KAN_MLP_CIFAR100()
        elif args.arch == 'kan_convnet_cifar100':
            model = KANConvNet_CIFAR100()
        elif args.arch == 'kan_convnet_cifar100_c':
            model = KANConvNet_CIFAR100_C()
        elif args.arch == 'kan_resnet_cifar100':
            model = KANResNet_CIFAR100(num_classes=100)
        elif args.arch == 'kagn_simple_cifar100_8_layer_v1':  ## in experiment for arxiv
            model = EightSimpleConvKAGN([32,  64, 128, 128, 192, 192, 256, 256],
                               num_classes=100, input_channels=3,
                               degree=3, groups=1, dropout=0.25, dropout_linear=0.5, l1_penalty=0.00000,
                               degree_out=3)
        elif args.arch == 'kagn_simple_cifar100_8_layer_v2':  ## in experiment for arxiv |||  also for ICML
            model = EightSimpleConvKAGN([64, 128, 256, 256, 384, 384, 512, 256],
                               num_classes=100, input_channels=3,
                               degree=3, groups=1, dropout=0.25, dropout_linear=0.5, l1_penalty=0.00000,
                               degree_out=3)
        
        elif args.arch == 'kagn_v2':  ## in experiment for ICML from https://huggingface.co/brivangl/vgg_kagn11_v2
            model = kagn_v2_cifar100()


    elif args.dataloader.dataset == 'tinyimagenet':
        if args.arch == 'kan_mlp_tinyimagenet':
            model = KAN_MLP_TinyImageNet_SM()
        elif args.arch == 'kan_mlp_tinyimagenet':
            model = KAN_MLP_TinyImageNet()
        elif args.arch == 'kan_convnet_tinyimagenet':
            model = KANConvNet_TinyImageNet()
        elif args.arch == 'kan_convnet_tinyimagenet_c':
            model = KANConvNet_TinyImageNet_C()
        elif args.arch == 'kan_resnet_tinyimagenet_sm':
            model = KANResNet_TinyImageNet_SM()  
        elif args.arch == 'kan_resnet_tinyimagenet':
            model = KANResNet_TinyImageNet()  
        
        elif args.arch == 'kagn_simple_tinyimagenet_8_layer_v1':  ## in experiment for arxiv
            model = EightSimpleConvKAGN([64, 128, 256, 256, 384, 384, 512, 512],
                               num_classes=200, input_channels=3,
                               degree=3, groups=1, dropout=0.25, dropout_linear=0.5, l1_penalty=0.00000,
                               degree_out=3)
            
        elif args.arch == 'kagn_simple_tinyimagenet_8_layer_v2':  ## in experiment for arxiv
            model = EightSimpleConvKAGN([128, 256, 512, 512, 640, 640, 768, 768],
                               num_classes=200, input_channels=3,
                               degree=3, groups=1, dropout=0.25, dropout_linear=0.5, l1_penalty=0.00000,
                               degree_out=3)
            
        elif args.arch == 'kagn_v2':  ## in experiment for ICML from https://huggingface.co/brivangl/vgg_kagn11_v2
            model = vggkagn(
                input_channels=3,
                num_classes=200,
                vgg_type='VGG11v2',      # or VGG11v4 if you prefer the larger variant
                width_scale=6,           # width scaling is more effective than extra depth:contentReference[oaicite:7]{index=7}
                degree=3,                # lower polynomial degree; scaling degree did not help performance
                groups=1,
                dropout=0.05,            # approximate full noise injection inside conv blocks
                dropout_linear=0.15,
                l1_decay=0,
                affine=True
            )
        elif args.arch == 'kagn_v4': # in experiment for ICML
            model = vggkagn(
                3,
                200,
                groups=1,
                degree=5, 
                dropout=0.15,
                l1_decay=0,
                dropout_linear=0.25,
                width_scale=2,
                vgg_type='VGG11v4',
                expected_feature_shape=(1, 1),
                affine=True
            )

    elif args.dataloader.dataset == 'imagenet':
        if args.arch == 'kagn_v2':  ## in experiment from https://huggingface.co/brivangl/vgg_kagn11_v2
            model = vggkagn(
                3,
                1000,
                groups=1,
                degree=5,
                dropout=0.15,
                l1_decay=0,
                dropout_linear=0.25,
                width_scale=2,
                vgg_type='VGG11v2',
                expected_feature_shape=(1, 1),
                affine=True
            )
        elif args.arch == 'kagn_v4':  ## in experiment from https://huggingface.co/brivangl/vgg_kagn11_v4 # in experiment for ICML
            model = vggkagn(
                3,
                1000,
                groups=1,
                degree=5,
                dropout=0.15,
                l1_decay=0,
                dropout_linear=0.25,
                width_scale=2,
                vgg_type='VGG11v4',
                expected_feature_shape=(1, 1),
                affine=True
            )

    elif args.dataloader.dataset == 'wine':
        if args.arch == 'kan_wine':
            model = KAN_Wine()

    elif args.dataloader.dataset == 'dry_bean':
        if args.arch == 'kan_dry_bean':
            model = KAN_DryBean()

    elif args.dataloader.dataset == 'jsc_openml':
        if args.arch == 'kan_jsc_openml':
            model = KAN_JSC_OpenML()

    elif args.dataloader.dataset == 'traffic_california':
        if args.arch == 'kan_traffic_3layer':      # paper-best, G=16
            model = KAN_Traffic(grid_size=16, hidden=True)
        elif args.arch == 'kan_traffic_3layer_g4':  # Table I baseline
            model = KAN_Traffic(grid_size=4, hidden=True)
        elif args.arch == 'kan_traffic_2layer':
            model = KAN_Traffic(grid_size=4, hidden=False)
        elif args.arch == 'mlp_traffic_3layer':
            model = MLP_Traffic_Baseline(four_layer=False)
        elif args.arch == 'mlp_traffic_4layer':
            model = MLP_Traffic_Baseline(four_layer=True)

    if model is None:
        logger.error('Model architecture `%s` for `%s` dataset is not supported' % (args.arch, args.dataloader.dataset))
        exit(-1)

    msg = 'Created `%s` model for `%s` dataset' % (args.arch, args.dataloader.dataset)
    msg += '\n          Use pre-trained model = %s' % args.pre_trained
    logger.info(msg)

    return model
