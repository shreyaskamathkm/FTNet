'''
Adapted from
https://github.com/zhanghang1989/ResNeSt
'''
"""ResNet variants"""
import os
import math
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from core.models.encoder.splat import SplAtConv2d

import os
os.environ['TORCH_HOME'] = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'pretrained_models'))
os.makedirs(os.environ['TORCH_HOME'], exist_ok=True)

root_pretrained_path = os.environ['TORCH_HOME']


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50',
           'resnet101', 'resnet152', 'resnet152_v1s', 'resnet101_v1s',
           'resnet50_v1s', 'resnest50', 'resnest101', 'resnest200', 'resnest269']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnest50': 'https://s3.us-west-1.wasabisys.com/resnest/torch/resnest50-528c19ca.pth',
    'resnest101': 'https://s3.us-west-1.wasabisys.com/resnest/torch/resnest101-22405ba7.pth',
    'resnest200': 'https://s3.us-west-1.wasabisys.com/resnest/torch/resnest200-75117900.pth',
    'resnest269': 'https://s3.us-west-1.wasabisys.com/resnest/torch/resnest269-0cc87c48.pth'
}


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, previous_dilation=1,
                 downsample=None, radix=1, cardinality=1, bottleneck_width=64,
                 avd=False, avd_first=False, is_first=False,
                 rectified_conv=False, rectify_avg=False,
                 norm_layer=None, dropblock_prob=0.0, last_gamma=False):

        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride,
                               dilation, dilation, bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, previous_dilation,
                               dilation=previous_dilation, bias=False)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
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


class DropBlock2D(object):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        return nn.functional.adaptive_avg_pool2d(inputs, 1).view(inputs.size(0), -1)


class Bottleneck(nn.Module):
    """ResNet Bottleneck
    """
    # pylint: disable=unused-argument
    expansion = 4

    def __init__(self, inplanes, planes, stride=1,
                 downsample=None, radix=1, cardinality=1, bottleneck_width=64,
                 avd=False, avd_first=False, dilation=1, previous_dilation=1, is_first=False,
                 rectified_conv=False, rectify_avg=False, norm_layer=None, dropblock_prob=0.0, last_gamma=False):
        super(Bottleneck, self).__init__()
        group_width = int(planes * (bottleneck_width / 64.)) * cardinality
        self.conv1 = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(group_width)
        self.dropblock_prob = dropblock_prob
        self.radix = radix
        self.avd = avd and (stride > 1 or is_first)
        self.avd_first = avd_first

        if self.avd:
            self.avd_layer = nn.AvgPool2d(3, stride, padding=1)
            stride = 1

        if dropblock_prob > 0.0:
            self.dropblock1 = DropBlock2D(dropblock_prob, 3)
            if radix == 1:
                self.dropblock2 = DropBlock2D(dropblock_prob, 3)
            self.dropblock3 = DropBlock2D(dropblock_prob, 3)

        if radix >= 1:
            self.conv2 = SplAtConv2d(
                group_width, group_width, kernel_size=3,
                stride=stride, padding=dilation,
                dilation=dilation, groups=cardinality, bias=False,
                radix=radix, rectify=rectified_conv,
                rectify_avg=rectify_avg,
                norm_layer=norm_layer,
                dropblock_prob=dropblock_prob)
        elif rectified_conv:
            from rfconv import RFConv2d
            self.conv2 = RFConv2d(
                group_width, group_width, kernel_size=3, stride=stride,
                padding=dilation, dilation=dilation,
                groups=cardinality, bias=False,
                average_mode=rectify_avg)
            self.bn2 = norm_layer(group_width)
        else:
            self.conv2 = nn.Conv2d(
                group_width, group_width, kernel_size=3, stride=stride,
                padding=dilation, dilation=dilation,
                groups=cardinality, bias=False)
            self.bn2 = norm_layer(group_width)

        self.conv3 = nn.Conv2d(
            group_width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * 4)

        if last_gamma:
            from torch.nn.init import zeros_
            zeros_(self.bn3.weight)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        if self.dropblock_prob > 0.0:
            out = self.dropblock1(out)
        out = self.relu(out)

        if self.avd and self.avd_first:
            out = self.avd_layer(out)

        out = self.conv2(out)
        if self.radix == 0:
            out = self.bn2(out)
            if self.dropblock_prob > 0.0:
                out = self.dropblock2(out)
            out = self.relu(out)

        if self.avd and not self.avd_first:
            out = self.avd_layer(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.dropblock_prob > 0.0:
            out = self.dropblock3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """ResNet Variants

    Parameters
    ----------
    block : Block
        Class for the residual block. Options are BasicBlockV1, BottleneckV1.
    layers : list of int
        Numbers of layers in each block
    classes : int, default 1000
        Number of classification classes.
    dilated : bool, default False
        Applying dilation strategy to pretrained ResNet yielding a stride-8 model,
        typically used in Semantic Segmentation.
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).

    Reference:

        - He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

        - Yu, Fisher, and Vladlen Koltun. "Multi-scale context aggregation by dilated convolutions."
    """
    # pylint: disable=unused-variable

    def __init__(self, block, layers, radix=1, groups=1, bottleneck_width=64,
                 num_classes=100, dilated=False, dilation=1,
                 deep_stem=False, stem_width=64, avg_down=False,
                 rectified_conv=False, rectify_avg=False,
                 avd=False, avd_first=False,
                 final_drop=0.0, dropblock_prob=0,
                 last_gamma=False, norm_layer=nn.BatchNorm2d, **kwargs):
        self.cardinality = groups
        self.bottleneck_width = bottleneck_width
        # ResNet-D params
        self.inplanes = stem_width * 2 if deep_stem else 64
        self.avg_down = avg_down
        self.last_gamma = last_gamma
        # ResNeSt params
        self.radix = radix
        self.avd = avd
        self.avd_first = avd_first

        super(ResNet, self).__init__()
        self.feature_list = []
        self.rectified_conv = rectified_conv
        self.rectify_avg = rectify_avg
        if rectified_conv:
            from rfconv import RFConv2d
            conv_layer = RFConv2d
        else:
            conv_layer = nn.Conv2d
        conv_kwargs = {'average_mode': rectify_avg} if rectified_conv else {}
        if deep_stem:
            self.conv1 = nn.Sequential(
                conv_layer(3, stem_width, kernel_size=3, stride=2, padding=1, bias=False, **conv_kwargs),
                norm_layer(stem_width),
                nn.ReLU(inplace=True),
                conv_layer(stem_width, stem_width, kernel_size=3, stride=1, padding=1, bias=False, **conv_kwargs),
                norm_layer(stem_width),
                nn.ReLU(inplace=True),
                conv_layer(stem_width, stem_width * 2, kernel_size=3, stride=1, padding=1, bias=False, **conv_kwargs),
            )
            self.feature_list.append(stem_width * 2)
        else:
            self.conv1 = conv_layer(3, 64, kernel_size=7, stride=2, padding=3,
                                    bias=False, **conv_kwargs)
            self.feature_list.append(64)

        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer, is_first=False)
        self.feature_list.append(self.inplanes)

        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        self.feature_list.append(self.inplanes)

        if dilated or dilation == 4:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
                                           dilation=2, norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob)
            self.feature_list.append(self.inplanes)

            self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                           dilation=4, norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob)
            self.feature_list.append(self.inplanes)

        elif dilation == 2:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                           dilation=1, norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob)
            self.feature_list.append(self.inplanes)

            self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                           dilation=2, norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob)
            self.feature_list.append(self.inplanes)

        else:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                           norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob)
            self.feature_list.append(self.inplanes)

            self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                           norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob)
            self.feature_list.append(self.inplanes)

        self.avgpool = GlobalAvgPool2d()
        self.drop = nn.Dropout(final_drop) if final_drop > 0.0 else None
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, norm_layer):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=None,
                    dropblock_prob=0.0, is_first=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            down_layers = []
            if self.avg_down:
                if dilation == 1:
                    down_layers.append(nn.AvgPool2d(kernel_size=stride, stride=stride,
                                                    ceil_mode=True, count_include_pad=False))
                else:
                    down_layers.append(nn.AvgPool2d(kernel_size=1, stride=1,
                                                    ceil_mode=True, count_include_pad=False))
                down_layers.append(nn.Conv2d(self.inplanes, planes * block.expansion,
                                             kernel_size=1, stride=1, bias=False))
            else:
                down_layers.append(nn.Conv2d(self.inplanes, planes * block.expansion,
                                             kernel_size=1, stride=stride, bias=False))
            down_layers.append(norm_layer(planes * block.expansion))
            downsample = nn.Sequential(*down_layers)

        layers = []
        if dilation == 1 or dilation == 2:
            layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                                radix=self.radix, cardinality=self.cardinality,
                                bottleneck_width=self.bottleneck_width,
                                avd=self.avd, avd_first=self.avd_first,
                                dilation=1, previous_dilation=dilation, is_first=is_first, rectified_conv=self.rectified_conv,
                                rectify_avg=self.rectify_avg,
                                norm_layer=norm_layer, dropblock_prob=dropblock_prob,
                                last_gamma=self.last_gamma))
        elif dilation == 4:
            layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                                radix=self.radix, cardinality=self.cardinality,
                                bottleneck_width=self.bottleneck_width,
                                avd=self.avd, avd_first=self.avd_first,
                                dilation=2, previous_dilation=dilation, is_first=is_first, rectified_conv=self.rectified_conv,
                                rectify_avg=self.rectify_avg,
                                norm_layer=norm_layer, dropblock_prob=dropblock_prob,
                                last_gamma=self.last_gamma))
        else:
            raise RuntimeError("=> unknown dilation size: {}".format(dilation))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                radix=self.radix, cardinality=self.cardinality,
                                bottleneck_width=self.bottleneck_width,
                                avd=self.avd, avd_first=self.avd_first,
                                dilation=dilation, previous_dilation=dilation, rectified_conv=self.rectified_conv,
                                rectify_avg=self.rectify_avg,
                                norm_layer=norm_layer, dropblock_prob=dropblock_prob,
                                last_gamma=self.last_gamma))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        x = torch.flatten(x, 1)
        if self.drop:
            x = self.drop(x)
        x = self.fc(x)

        return x


def resnet18(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], radix=0, **kwargs)
    if pretrained:
        from core.models.segbase import check_mismatch

        pretrained_dict = model_zoo.load_url(model_urls['resnet18'])
        pretrained_dict = check_mismatch(model, pretrained_dict)
        model.load_state_dict(pretrained_dict, strict=False)

    return model


def resnet34(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], radix=0, **kwargs)
    if pretrained:
        from core.models.segbase import check_mismatch

        pretrained_dict = model_zoo.load_url(model_urls['resnet34'])
        pretrained_dict = check_mismatch(model, pretrained_dict)
        model.load_state_dict(pretrained_dict, strict=False)
    return model


def resnet50(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], radix=0, **kwargs)
    # print_network(model)
    if pretrained:
        from core.models.segbase import check_mismatch
        pretrained_dict = model_zoo.load_url(model_urls['resnet50'])
        pretrained_dict = check_mismatch(model, pretrained_dict)
        model.load_state_dict(pretrained_dict, strict=False)
    return model


def resnet101(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], radix=0, **kwargs)
    if pretrained:
        from core.models.segbase import check_mismatch
        pretrained_dict = model_zoo.load_url(model_urls['resnet101'])
        pretrained_dict = check_mismatch(model, pretrained_dict)
        model.load_state_dict(pretrained_dict, strict=False)
    return model


def resnet152(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 8, 36, 3], radix=0, **kwargs)
    if pretrained:
        from core.models.segbase import check_mismatch
        pretrained_dict = model_zoo.load_url(model_urls['resnet152'])
        pretrained_dict = check_mismatch(model, pretrained_dict)
        model.load_state_dict(pretrained_dict, strict=False)
    return model


def resnet50_v1s(pretrained=False, root=root_pretrained_path, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], deep_stem=True, radix=0, **kwargs)
    if pretrained:
        from core.models.segbase import check_mismatch
        pretrained_dict = torch.load(os.path.join(root, 'resnet50_v1s.pth'))
        pretrained_dict = check_mismatch(model, pretrained_dict)
        model.load_state_dict(pretrained_dict, strict=False)
    return model


def resnet101_v1s(pretrained=False, root=root_pretrained_path, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], deep_stem=True, radix=0, **kwargs)
    if pretrained:
        from core.models.segbase import check_mismatch
        pretrained_dict = torch.load(os.path.join(root, 'resnet101_v1s.pth'))
        pretrained_dict = check_mismatch(model, pretrained_dict)
        model.load_state_dict(pretrained_dict, strict=False)
    return model


def resnet152_v1s(pretrained=False, root=root_pretrained_path, **kwargs):
    model = ResNet(Bottleneck, [3, 8, 36, 3], deep_stem=True, radix=0, **kwargs)
    if pretrained:
        from core.models.segbase import check_mismatch
        pretrained_dict = torch.load(os.path.join(root, 'resnet152_v1s.pth'))
        pretrained_dict = check_mismatch(model, pretrained_dict)
        model.load_state_dict(pretrained_dict, strict=False)
    return model


def resnest50(pretrained=False, root=root_pretrained_path, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=32, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    if pretrained:
        from core.models.segbase import check_mismatch
        pretrained_dict = torch.hub.load_state_dict_from_url(model_urls['resnest50'], progress=True, check_hash=True)
        pretrained_dict = check_mismatch(model, pretrained_dict)
        model.load_state_dict(pretrained_dict, strict=False)
    return model


def resnest101(pretrained=False, root=root_pretrained_path, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    if pretrained:
        from core.models.segbase import check_mismatch
        pretrained_dict = torch.hub.load_state_dict_from_url(model_urls['resnest101'], progress=True, check_hash=True)
        pretrained_dict = check_mismatch(model, pretrained_dict)
        model.load_state_dict(pretrained_dict, strict=False)
    return model


def resnest200(pretrained=False, root=root_pretrained_path, **kwargs):
    model = ResNet(Bottleneck, [3, 24, 36, 3],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    if pretrained:
        from core.models.segbase import check_mismatch
        pretrained_dict = torch.hub.load_state_dict_from_url(model_urls['resnest200'], progress=True, check_hash=True)
        pretrained_dict = check_mismatch(model, pretrained_dict)
        model.load_state_dict(pretrained_dict, strict=False)

    return model


def resnest269(pretrained=False, root=root_pretrained_path, **kwargs):
    model = ResNet(Bottleneck, [3, 30, 48, 8],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    if pretrained:
        from core.models.segbase import check_mismatch
        pretrained_dict = torch.hub.load_state_dict_from_url(model_urls['resnest269'], progress=True, check_hash=True)
        pretrained_dict = check_mismatch(model, pretrained_dict)
        model.load_state_dict(pretrained_dict, strict=False)
    return model


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: {} '.format(int(num_params)))
    print('Total number of parameters: {:.3f} M'.format(int(num_params) / 1000**2))


if __name__ == '__main__':

    import os
    from os import path, sys
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    sys.path.append(os.path.join(path.dirname(path.dirname(path.abspath(__file__))), '..'))
    from ptflops import get_model_complexity_info
    img = torch.randint(3, 5, (2, 3, 512, 512)).float()
    model = resnet50(pretrained_base=False, dilated=True, dilation=4)
    # model = model.eval()

    outputs = model(img)
    print_network(model)
    macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True,
                                             print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
