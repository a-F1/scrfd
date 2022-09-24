import torch.nn as nn

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self,inplanes,planes,
                stride=1,dilation=1,
                downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=inplanes,
            out_channels=planes,
            kernel_size = 3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        """Forward function."""

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity

        out = self.relu(out)
        return out


class SCRFD_10G(nn.Module):
    def __init__(self, block=BasicBlock,
                 stage_blocks=[3, 4, 2, 3],
                 stage_planes=[56, 88, 88, 224],
                 in_channels = 3,base_channels=56,
                 strides=[1,2,2,2],
                 out_stage=[1, 2, 3,4],
                 avg_down=True
        ) -> None:
        super(SCRFD_10G,self).__init__()
        self.in_channels = in_channels
        self.stem_channels = base_channels
        self.base_channels = base_channels
        self.avg_down = avg_down
        self.out_stage = out_stage
        self.deep_stem = self._make_stem_layer(self.in_channels,self.stem_channels)

        self.layer1 = self._make_res_layer(block, self.stem_channels, stage_planes[0],stage_blocks[0],stride=strides[0],avg_down=self.avg_down)
        self.layer2 = self._make_res_layer(block, stage_planes[0]*block.expansion, stage_planes[1],stage_blocks[1], stride=strides[1],avg_down=self.avg_down)
        self.layer3 = self._make_res_layer(block, stage_planes[1]*block.expansion, stage_planes[2],stage_blocks[2], stride=strides[2],avg_down=self.avg_down)
        self.layer4 = self._make_res_layer(block, stage_planes[2]*block.expansion, stage_planes[3],stage_blocks[3], stride=strides[3],avg_down=self.avg_down)

    def _make_res_layer(self, block, inplanes,planes, num_blocks, stride=1,avg_down=False):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = []
            conv_stride = stride
            if avg_down:
                conv_stride = 1
                downsample.append(nn.AvgPool2d(kernel_size=stride,stride=stride,ceil_mode=True,count_include_pad=False))
            downsample.extend([
                nn.Conv2d(inplanes,planes*block.expansion,kernel_size=1,stride=conv_stride,bias=False),
                nn.BatchNorm2d(planes*block.expansion)
            ])
            downsample = nn.Sequential(*downsample)

        layers = []
        layers.append(block(inplanes, planes, stride, 1,downsample))
        inplanes = planes * block.expansion
        for _ in range(1,num_blocks):
            layers.append(block(inplanes,planes))
        return nn.Sequential(*layers)

    def _make_stem_layer(self, in_channels, stem_channels):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                stem_channels // 2,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(stem_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                stem_channels // 2,
                stem_channels // 2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(stem_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                stem_channels // 2,
                stem_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
    def forward(self,x):
        x = self.deep_stem(x)
        outs = []
        for i in range(1,5):
            layer = getattr(self,"layer{}".format(i))
            x = layer(x)
            if i in self.out_stage:
                outs.append(x)
        return tuple(outs)

if __name__ == "__main__":
    model = SCRFD_10G()
    print(model)