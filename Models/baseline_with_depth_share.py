import torch
import torch.nn as nn
import torch.nn.functional as F

class BiSalNet(nn.Module):
    def __init__(self):
        super(BiSalNet, self).__init__()
        self.context_path = VGG16_bn(pretrained=True)
        # self.global_context = PyramidPooling(512, 256)
        self.prepare = nn.ModuleList([
                convbnrelu_double(in_channel=512, out_channel=256, k=1, s=1, p=0, relu=False),
                convbnrelu_double(in_channel=512, out_channel=256, k=1, s=1, p=0, relu=False),
                convbnrelu_double(in_channel=256, out_channel=128, k=1, s=1, p=0, relu=False),
                convbnrelu_double(in_channel=128, out_channel=64, k=1, s=1, p=0, relu=False),
                convbnrelu_double(in_channel=64, out_channel=32, k=1, s=1, p=0, relu=False)
                ])
        self.fuse = nn.ModuleList([
                convbnrelu(in_channel=256, out_channel=256, k=1, s=1, p=0, relu=True),
                convbnrelu(in_channel=256, out_channel=128, k=1, s=1, p=0, relu=True),
                convbnrelu(in_channel=128, out_channel=64, k=1, s=1, p=0, relu=True),
                convbnrelu(in_channel=64, out_channel=32, k=1, s=1, p=0, relu=True),
                convbnrelu(in_channel=32, out_channel=16, k=1, s=1, p=0, relu=True)
                ])
        self.heads = nn.ModuleList([
                SalHead(in_channel=256),
                SalHead(in_channel=128),
                SalHead(in_channel=64),
                SalHead(in_channel=32),
                SalHead(in_channel=16)
                ])

    def forward(self, x, depth): # (3, 1)
        stages_rgb = self.context_path(x, depth_path=False)
        rgb_stage1, rgb_stage2, rgb_stage3, rgb_stage4, rgb_stage5 = stages_rgb
        # (64, 1)    (128, 1/2)  (256, 1/4)  (512, 1/8)  (512, 1/16)
        # rgb_stage6 = self.global_context(rgb_stage5)                                            # (256, 1/16)

        stages_depth = self.context_path(depth, depth_path=True)
        depth_stage1, depth_stage2, depth_stage3, depth_stage4, depth_stage5 = stages_depth
        # (64, 1)      (128, 1/2)    (256, 1/4)    (512, 1/8)    (512, 1/16)
        
        rgb_stage5_prepare = self.prepare[0](rgb_stage5, depth_path=False)                      # (256, 1/16)
        depth_stage5_prepare = self.prepare[0](depth_stage5, depth_path=True)                   # (256, 1/16)
        fused_stage1 = self.fuse[0](rgb_stage5_prepare + depth_stage5_prepare)                  # (256, 1/16)
        refined1 = interpolate(fused_stage1, rgb_stage4.size()[2:])                             # (256, 1/8)
        
        rgb_stage4_prepare = self.prepare[1](rgb_stage4, depth_path=False)                      # (256, 1/8)
        depth_stage4_prepare = self.prepare[1](depth_stage4, depth_path=True)                   # (256, 1/8)
        fused_stage2 = self.fuse[1](rgb_stage4_prepare + depth_stage4_prepare + refined1)       # (128, 1/8)
        refined2 = interpolate(fused_stage2, rgb_stage3.size()[2:])                             # (128, 1/4)
        
        rgb_stage3_prepare = self.prepare[2](rgb_stage3, depth_path=False)                      # (128, 1/4)
        depth_stage3_prepare = self.prepare[2](depth_stage3, depth_path=True)                   # (128, 1/4)
        fused_stage3 = self.fuse[2](rgb_stage3_prepare + depth_stage3_prepare + refined2)       # (64, 1/4)
        refined3 = interpolate(fused_stage3, rgb_stage2.size()[2:])                             # (64, 1/2)
        
        rgb_stage2_prepare = self.prepare[3](rgb_stage2, depth_path=False)                      # (64, 1/2)
        depth_stage2_prepare = self.prepare[3](depth_stage2, depth_path=True)                   # (64, 1/2)
        fused_stage4 = self.fuse[3](rgb_stage2_prepare + depth_stage2_prepare + refined3)       # (32, 1/2)
        refined4 = interpolate(fused_stage4, rgb_stage1.size()[2:])	                            # (32, 1)
        
        rgb_stage1_prepare = self.prepare[4](rgb_stage1, depth_path=False)                      # (32, 1)
        depth_stage1_prepare = self.prepare[4](depth_stage1, depth_path=True)                   # (32, 1)
        fused_stage5 = self.fuse[4](rgb_stage1_prepare + depth_stage1_prepare + refined4)       # (16, 1)

        output_side1 = interpolate(self.heads[0](fused_stage1), x.size()[2:])
        output_side2 = interpolate(self.heads[1](fused_stage2), x.size()[2:])
        output_side3 = interpolate(self.heads[2](fused_stage3), x.size()[2:])
        output_side4 = interpolate(self.heads[3](fused_stage4), x.size()[2:])
        output_main  = self.heads[4](fused_stage5)
        
        return output_main, output_side1, output_side2, output_side3, output_side4, [stages_rgb, stages_depth]

interpolate = lambda x, size: F.interpolate(x, size=size, mode="bilinear", align_corners=True)

class convbnrelu(nn.Module):
    def __init__(self, in_channel, out_channel, k, s, p, d=1, bias=False, bn=True, relu=True):
        super(convbnrelu, self).__init__()
        conv = [nn.Conv2d(in_channel, out_channel, k, stride=s, padding=p, dilation=d, bias=bias)]
        if bn:
            conv.append(nn.BatchNorm2d(out_channel))
        if relu:
            conv.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*conv)
            
    def forward(self, x):
        return self.conv(x)
        
class convbnrelu_double(nn.Module):
    def __init__(self, in_channel, out_channel, k, s, p, d=1, bias=False, relu=True):
        super(convbnrelu_double, self).__init__()
        self.use_relu = relu
        self.conv = nn.Conv2d(in_channel, out_channel, k, stride=s, padding=p, dilation=d, bias=bias)
        self.bn = nn.ModuleList([nn.BatchNorm2d(out_channel), nn.BatchNorm2d(out_channel)])
            
    def forward(self, x, depth_path=False):
        idx = 1 if depth_path else 0
        x = self.conv(x)
        out = self.bn[idx](x)
        if self.use_relu:
            out = F.relu(out, inplace=True)
        return out
            
class SalHead(nn.Module):
    def __init__(self, in_channel):
        super(SalHead, self).__init__()
        self.conv = nn.Sequential(
                nn.Dropout2d(p=0.1),
                nn.Conv2d(in_channel, 1, 1, stride=1, padding=0),
                nn.Sigmoid()
                )

    def forward(self, x):
        return self.conv(x) 

class PyramidPooling(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(PyramidPooling, self).__init__()
        hidden_channel = int(in_channel / 4)
        self.conv1 = convbnrelu(in_channel, hidden_channel, k=1, s=1, p=0)
        self.conv2 = convbnrelu(in_channel, hidden_channel, k=1, s=1, p=0)
        self.conv3 = convbnrelu(in_channel, hidden_channel, k=1, s=1, p=0)
        self.conv4 = convbnrelu(in_channel, hidden_channel, k=1, s=1, p=0)
        self.out = convbnrelu(in_channel*2, out_channel, k=1, s=1, p=0)

    def forward(self, x):
        size = x.size()[2:]
        feat1 = interpolate(self.conv1(F.adaptive_avg_pool2d(x, 1)), size)
        feat2 = interpolate(self.conv2(F.adaptive_avg_pool2d(x, 2)), size)
        feat3 = interpolate(self.conv3(F.adaptive_avg_pool2d(x, 3)), size)
        feat4 = interpolate(self.conv4(F.adaptive_avg_pool2d(x, 6)), size)
        x = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)
        x = self.out(x)
        return x
    
class VGG16_bn(nn.Module):
    def __init__(self, pretrained=False, path="./resnet_pretrained/vgg16_bn_doubleBN.pth"):
        super(VGG16_bn, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.bn1_1 = nn.ModuleList([nn.BatchNorm2d(64), nn.BatchNorm2d(64)])
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.bn1_2 = nn.ModuleList([nn.BatchNorm2d(64), nn.BatchNorm2d(64)])
        
        self.conv2_1 = nn.Conv2d(64, 128, 3, 1, 1)
        self.bn2_1 = nn.ModuleList([nn.BatchNorm2d(128), nn.BatchNorm2d(128)])
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.bn2_2 = nn.ModuleList([nn.BatchNorm2d(128), nn.BatchNorm2d(128)])

        self.conv3_1 = nn.Conv2d(128, 256, 3, 1, 1)
        self.bn3_1 = nn.ModuleList([nn.BatchNorm2d(256), nn.BatchNorm2d(256)])
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.bn3_2 = nn.ModuleList([nn.BatchNorm2d(256), nn.BatchNorm2d(256)])
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 1)
        self.bn3_3 = nn.ModuleList([nn.BatchNorm2d(256), nn.BatchNorm2d(256)])

        self.conv4_1 = nn.Conv2d(256, 512, 3, 1, 1)
        self.bn4_1 = nn.ModuleList([nn.BatchNorm2d(512), nn.BatchNorm2d(512)])
        self.conv4_2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.bn4_2 = nn.ModuleList([nn.BatchNorm2d(512), nn.BatchNorm2d(512)])
        self.conv4_3 = nn.Conv2d(512, 512, 3, 1, 1)
        self.bn4_3 = nn.ModuleList([nn.BatchNorm2d(512), nn.BatchNorm2d(512)])

        self.conv5_1 = nn.Conv2d(512, 512, 3, 1, 1)
        self.bn5_1 = nn.ModuleList([nn.BatchNorm2d(512), nn.BatchNorm2d(512)])
        self.conv5_2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.bn5_2 = nn.ModuleList([nn.BatchNorm2d(512), nn.BatchNorm2d(512)])
        self.conv5_3 = nn.Conv2d(512, 512, 3, 1, 1)
        self.bn5_3 = nn.ModuleList([nn.BatchNorm2d(512), nn.BatchNorm2d(512)])

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2, 2)
        
        if pretrained:
            self.load_state_dict(torch.load(path), strict=True)
        
    def forward(self, x, depth_path=False):
        idx = 1 if depth_path else 0
        
        conv = self.conv1_1(x)
        conv = self.conv1_2(self.relu(self.bn1_1[idx](conv)))
        out1 = self.relu(self.bn1_2[idx](conv))
        
        conv = self.conv2_1(self.maxpool(out1))
        conv = self.conv2_2(self.relu(self.bn2_1[idx](conv)))
        out2 = self.relu(self.bn2_2[idx](conv))
        
        conv = self.conv3_1(self.maxpool(out2))
        conv = self.conv3_2(self.relu(self.bn3_1[idx](conv)))
        conv = self.conv3_3(self.relu(self.bn3_2[idx](conv)))
        out3 = self.relu(self.bn3_3[idx](conv))
        
        conv = self.conv4_1(self.maxpool(out3))
        conv = self.conv4_2(self.relu(self.bn4_1[idx](conv)))
        conv = self.conv4_3(self.relu(self.bn4_2[idx](conv)))
        out4 = self.relu(self.bn4_3[idx](conv))
        
        conv = self.conv5_1(self.maxpool(out4))
        conv = self.conv5_2(self.relu(self.bn5_1[idx](conv)))
        conv = self.conv5_3(self.relu(self.bn5_2[idx](conv)))
        out5 = self.relu(self.bn5_3[idx](conv))
        
        return out1, out2, out3, out4, out5

if __name__ == "__main__":
    x = torch.randn(2, 3, 224, 224)
    model = BiSalNet()
    am, a1, a2, a3, a4 = model(x, x)
    print(am.shape)
    print(a1.shape)
    print(a2.shape)
    print(a3.shape)
    print(a4.shape)
