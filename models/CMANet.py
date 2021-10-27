import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet50
from models.ResNet import ResNet_rgb, ResNet_depth

resnet_2D = resnet50(pretrained=True)

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
class TransBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, upsample=None, **kwargs):
        super(TransBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, inplanes)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        if upsample is not None and stride != 1:
            self.conv2 = nn.ConvTranspose2d(inplanes, planes,
                                            kernel_size=3, stride=stride, padding=1,
                                            output_padding=1, bias=False)
        else:
            self.conv2 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            residual = self.upsample(x)

        out += residual
        out = self.relu(out)

        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)


class AttenBlock(nn.Module):
    def __init__(self, channel=32):
        super(AttenBlock, self).__init__()
        self.atten_channel = ChannelAttention(channel)
        #self.atten_channel = eca_layer()

    def forward(self, x, y):
        attention = y.mul(self.atten_channel(x))
        out = y + attention

        return out


class CoAttenBlock(nn.Module):
    def __init__(self, channel=32):
        super(CoAttenBlock, self).__init__()
        self.concaL = nn.Conv2d(in_channels=channel * 2, out_channels=channel, kernel_size=1)
        self.concaR = nn.Conv2d(in_channels=channel * 2, out_channels=channel, kernel_size=1)
        self.concaF = nn.Conv2d(in_channels=channel * 2, out_channels=channel, kernel_size=1)
        self.weightL = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1)
        self.weightR = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1)
        self.weightF = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1)
        self.gateL = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=1, kernel_size=1), nn.Sigmoid(), )
        self.gateR = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=1, kernel_size=1), nn.Sigmoid(), )
        self.out = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1)

    def forward(self, xlh, xll, xrh, xrl):
        bs, ch, hei, wei = xlh.size()

        # multi-level fusion
        xL = self.concaL(torch.cat([xlh, xll], dim=1))
        xR = self.concaR(torch.cat([xrh, xrl], dim=1))

        # corsely fusion
        xF = self.concaF(torch.cat([xL, xR], dim=1))

        # inspired by k, q, v in self-attention
        valL = self.weightL(xL)
        valR = self.weightR(xR)
        valF = self.weightF(xF)

        # multi-modal fusion
        valL_reshape = torch.flatten(valL, start_dim=2, end_dim=3)
        valR_reshape = torch.flatten(valR, start_dim=2, end_dim=3)
        valF_reshape = torch.flatten(valF, start_dim=2, end_dim=3)

        Similarity = torch.matmul(valR_reshape.permute(0, 2, 1), valL_reshape)
        func1 = F.softmax(Similarity, dim=1)
        func2 = F.softmax(Similarity, dim=2)
        #debug = torch.sum(func1[0,0,:])
        Attention1 = torch.matmul(func1, valF_reshape.permute(0, 2, 1))
        Attention1 = Attention1.permute(0, 2, 1)
        Attention1 = Attention1.reshape([bs, ch, hei, wei])
        AttenGate1 = self.gateL(Attention1)
        Attention1 = Attention1.mul(AttenGate1)
        Attention2 = torch.matmul(func2, valF_reshape.permute(0, 2, 1))
        Attention2 = Attention2.permute(0, 2, 1)
        Attention2 = Attention2.reshape([bs, ch, hei, wei])
        AttenGate2 = self.gateR(Attention2)
        Attention2 = Attention2.mul(AttenGate2)

        out_L = xL + Attention1
        out_R = xR + Attention2

        return out_L, out_R


class BasicResConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicResConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.conv(x)
        x = self.bn(x)
        out = x + residual
       # out = self.relu(out)

        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x=max_out
        x = self.conv1(x)
        return self.sigmoid(x)


class aggregation_MA(nn.Module):
    def __init__(self):
        super(aggregation_MA, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.CoA_1 = CoAttenBlock()
        self.CoA_2 = CoAttenBlock()

        # multi-supervision
        self.out_conv_L = nn.Conv2d(32 * 1, 1, kernel_size=1, stride=1, bias=True)
        self.out_conv_R = nn.Conv2d(32 * 1, 1, kernel_size=1, stride=1, bias=True)

        # Components of PTM module
        self.inplanes = 32
        self.deconv1 = self._make_transpose(TransBasicBlock, 32, 3, stride=2)
        self.deconv2 = self._make_transpose(TransBasicBlock, 32, 3, stride=2)
        self.deconv3 = self._make_transpose(TransBasicBlock, 32, 3, stride=2)
        self.agant1 = self._make_agant_layer(32 * 2, 32)
        self.agant2 = self._make_agant_layer(32, 32)
        self.agant3 = self._make_agant_layer(32, 32)
        self.out_conv = nn.Conv2d(32, 1, kernel_size=1, stride=1, bias=True)

    def forward(self, l1, l2, l3, r1, r2, r3):
        l1 = self.upsample(l1)
        r1 = self.upsample(r1)
        add_1_out_L, add_1_out_R = self.CoA_1(l1, l2, r1, r2)

        add_1_out_L = self.upsample(add_1_out_L)
        add_1_out_R = self.upsample(add_1_out_R)

        add_2_out_L, add_2_out_R = self.CoA_2(add_1_out_L, l3, add_1_out_R, r3)

        out_L = self.upsample8(self.out_conv_L(add_2_out_L))
        out_R = self.upsample8(self.out_conv_R(add_2_out_R))
        add_2_out = torch.cat([add_2_out_L, add_2_out_R], dim=1)

        out = self.agant1(add_2_out)
        out = self.deconv1(out)
        out = self.agant2(out)
        out = self.deconv2(out)
        out = self.agant3(out)
        out = self.deconv3(out)
        out = self.out_conv(out)

        return out, out_L, out_R

    def _make_agant_layer(self, inplanes, planes):
        layers = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        return layers

    def _make_transpose(self, block, planes, blocks, stride=1):
        upsample = None
        if stride != 1:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.inplanes, planes,
                                   kernel_size=2, stride=stride,
                                   padding=0, bias=False),
                nn.BatchNorm2d(planes),
            )
        elif self.inplanes != planes:
            upsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []
        for i in range(1, blocks):
            layers.append(block(self.inplanes, self.inplanes))
        layers.append(block(self.inplanes, planes, stride, upsample))
        self.inplanes = planes

        return nn.Sequential(*layers)


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
      #  x = self.relu(x)
        return x


class RFB(nn.Module):
    # RFB-like multi-scale module
    def __init__(self, in_channel, out_channel):
        super(RFB, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


class CMANet(nn.Module):
  def __init__(self):
    super(CMANet, self).__init__()

    channel_decoder = 32

    # encoder
    self.encoder_rgb = ResNet_rgb()
    self.encoder_depth = ResNet_depth()

    self.rfb_rgb_3 = RFB(512, channel_decoder)
    self.rfb_rgb_4 = RFB(1024, channel_decoder)
    self.rfb_rgb_5 = RFB(2048, channel_decoder)
    self.rfb_depth_3 = RFB(512, channel_decoder)
    self.rfb_depth_4 = RFB(1024, channel_decoder)
    self.rfb_depth_5 = RFB(2048, channel_decoder)
    self.agg = aggregation_MA()

    if self.training: self.initialize_weights()

  def forward(self, rgb, depth):
    # guidance of high-level encoding of focus stacking
    E_rgb5, E_rgb4, E_rgb3, _, _ = self.encoder_rgb.forward(rgb)
    E_depth5, E_depth4, E_depth3, _, _ = self.encoder_depth.forward(depth)

    d3_rgb = self.rfb_rgb_3(E_rgb3)
    d4_rgb = self.rfb_rgb_4(E_rgb4)
    d5_rgb = self.rfb_rgb_5(E_rgb5)

    d3_depth = self.rfb_depth_3(E_depth3)
    d4_depth = self.rfb_depth_4(E_depth4)
    d5_depth = self.rfb_depth_5(E_depth5)

    pred_fuse, pred_L, pred_R = self.agg(d5_rgb, d4_rgb, d3_rgb, d5_depth, d4_depth, d3_depth)

    return pred_fuse, pred_L, pred_R

  def initialize_weights(self):
    pretrained_dict = resnet_2D.state_dict()
    all_params = {}
    for k, v in self.encoder_rgb.state_dict().items():
      if k in pretrained_dict.keys():
        v = pretrained_dict[k]
        all_params[k] = v
    self.encoder_rgb.load_state_dict(all_params)

    all_params_depth = {}
    for k, v in self.encoder_depth.state_dict().items():
        if k in pretrained_dict.keys():
            if k == 'conv1.weight':
                continue
            v = pretrained_dict[k]
            all_params_depth[k] = v
    self.encoder_depth.load_state_dict(all_params_depth, strict=False)
