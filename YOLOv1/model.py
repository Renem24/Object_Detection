
import torch
import torch.nn as nn


"""
Tuple - (kernel_size, filters, stride, padding)
"M"     -   simply maxpooling with stride 2x2 and kernel 2x2

"""
architecture_config = [
    ## tuple
    (7, 64, 2, 3),  # 7x7x64(W x H x C_output), stride 2, C_input 3
    ## str
    "M",            # Max Pooling 
    (3, 192, 1, 1), # 3x3x192(W x H x C_output), stride 2, C_input 1
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    ## list
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]



class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)
    
    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))


class Yolov1(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super(Yolov1, self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs(**kwargs)
    
    ### Forward Pass(순전파)
    def forward(self, x):
        x = self.darknet(x)
        # fcs : fully connected layer들을 nn.Sequential로 묶어 둔 모듈
        return self.fcs(torch.flatten(x, start_dim=1))
    
    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels
        
        ## 위에서 작성한 모델 설계도(architecture_config)에 대한, 각각의 layer 구현
        for x in architecture:
            if type(x) == tuple:
                layers += [
                    CNNBlock(
                        ## def __init__(self, in_channels, out_channels, **kwargs):
                        in_channels, x[1], kernel_size=x[0], stride=x[2], padding=x[3],
                    )
                ]
                in_channels = x[1]
            
            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]
            
            elif type(x) == list:
                conv1 = x[0]
                conv2 = x[1]
                num_repeats = x[2]
                
                for _ in range(num_repeats):
                    layers += [
                        CNNBlock(
                            in_channels,
                            conv1[1],
                            kernel_size=conv1[0],
                            stride=conv1[2],
                            padding=conv1[3],
                        )
                    ]
                    layers += [
                        CNNBlock(
                            conv1[1],
                            conv2[1],
                            kernel_size=conv2[0],
                            stride=conv2[2],
                            padding=conv2[3],
                        )
                    ]
                    in_channels = conv2[1]
                    
        return nn.Sequential(*layers)

    def _create_fcs(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes
        # S x S 개의 grids
        # grid cell당, B 개의 boxes
        # gird cell당, 전체 class의 종류에 대한 개별 확률(class probability)
        
        ## In original paper this should be
        # nn.Linear(1024*S*S, 4096),
        # nn.LeakyReLU(0.1),
        # nn.Linear(4096, S*S*(B*5+C))
        
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 496),
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(496, S * S * (C + B * 5))
        )




## nn.Sequential(*layers)이란?
# model = nn.Sequential(
#     nn.Linear(in_features=10, out_features=20),  # First linear layer
#     nn.ReLU(),                                  # Activation function
#     nn.Linear(in_features=20, out_features=1)   # Second linear layer
# )