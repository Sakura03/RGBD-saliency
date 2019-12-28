import torch
import torch.nn as nn
import torch.nn.functional as F

class BiSalNet(nn.Module):
    def __init__(self):
        super(BiSalNet, self).__init__()
        self.context_path = VGG16_bn(pretrained=True)
        self.global_context = PyramidPooling(512, 256)
        self.prepare = nn.ModuleList([
                convbnrelu(in_channel=512, out_channel=256, k=1, s=1, p=0, relu=False),
                convbnrelu(in_channel=512, out_channel=256, k=1, s=1, p=0, relu=False),
                convbnrelu(in_channel=256, out_channel=128, k=1, s=1, p=0, relu=False),
                convbnrelu(in_channel=128, out_channel=64, k=1, s=1, p=0, relu=False),
                convbnrelu(in_channel=64, out_channel=32, k=1, s=1, p=0, relu=False)
                ])
        self.fuse = nn.ModuleList([
                convbnrelu(in_channel=256, out_channel=256, k=3, s=1, p=1, relu=True),
                convbnrelu(in_channel=256, out_channel=128, k=3, s=1, p=2, d=2, relu=True),
                convbnrelu(in_channel=128, out_channel=64, k=3, s=1, p=2, d=2, relu=True),
                convbnrelu(in_channel=64, out_channel=32, k=5, s=1, p=4, d=2, relu=True),
                convbnrelu(in_channel=32, out_channel=16, k=5, s=1, p=4, d=2, relu=True)
                ])
        self.heads = nn.ModuleList([
                SalHead(in_channel=256),
                SalHead(in_channel=128),
                SalHead(in_channel=64),
                SalHead(in_channel=32),
                SalHead(in_channel=16)
                ])

    def forward(self, x): # (3, 1)
        size = x.size()[2:]
        ct_stage1, ct_stage2, ct_stage3, ct_stage4, ct_stage5 = self.context_path(x)
        #(64, 1)  (128, 1/2) (256, 1/4) (512, 1/8) (512, 1/16)
        ct_stage6 = self.global_context(ct_stage5)                                # (256, 1/16)
        
        fused_stage1 = self.fuse[0](self.prepare[0](ct_stage5) + ct_stage6)       # (256, 1/16)
        refined1 = interpolate(fused_stage1, ct_stage4.size()[2:])                # (256, 1/8)
        
        fused_stage2 = self.fuse[1](self.prepare[1](ct_stage4) + refined1)        # (128, 1/8)
        refined2 = interpolate(fused_stage2, ct_stage3.size()[2:]) 		            # (128, 1/4)
        
        fused_stage3 = self.fuse[2](self.prepare[2](ct_stage3) + refined2)        # (64, 1/4)
        refined3 = interpolate(fused_stage3, ct_stage2.size()[2:]) 		            # (64, 1/2)
        
        fused_stage4 = self.fuse[3](self.prepare[3](ct_stage2) + refined3)        # (32, 1/2)
        refined4 = interpolate(fused_stage4, ct_stage1.size()[2:])		            # (32, 1)
        
        fused_stage5 = self.fuse[4](self.prepare[4](ct_stage1) + refined4)        # (16, 1)

        output_side1 = interpolate(self.heads[0](fused_stage1), x.size()[2:])
        output_side2 = interpolate(self.heads[1](fused_stage2), x.size()[2:])
        output_side3 = interpolate(self.heads[2](fused_stage3), x.size()[2:])
        output_side4 = interpolate(self.heads[3](fused_stage4), x.size()[2:])
        output_main  = self.heads[4](fused_stage5)
        return output_main, output_side1, output_side2, output_side3, output_side4

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
    def __init__(self, pretrained=False, path="resnet_pretrained/vgg16_bn.pth"):
        super(VGG16_bn, self).__init__()
        self.layer1 = nn.Sequential(
                nn.Conv2d(3, 64, 3, 1, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, 1, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
                )
        
        self.layer2 = nn.Sequential(
                nn.MaxPool2d(2, 2),
                nn.Conv2d(64, 128, 3, 1, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, 1, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
                )
        
        self.layer3 = nn.Sequential(
                nn.MaxPool2d(2, 2),
                nn.Conv2d(128, 256, 3, 1, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, 3, 1, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, 3, 1, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
                )
        
        self.layer4 = nn.Sequential(
                nn.MaxPool2d(2, 2),
                nn.Conv2d(256, 512, 3, 1, 1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, 3, 1, 1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, 3, 1, 1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
                )
        
        self.layer5 = nn.Sequential(
                nn.MaxPool2d(2, 2),
                nn.Conv2d(512, 512, 3, 1, 1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, 3, 1, 1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, 3, 1, 1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
                )
        
        if pretrained:
            self.load_state_dict(torch.load(path), strict=False)
        
    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        return out1, out2, out3, out4, out5

if __name__ == "__main__":
    x = torch.randn(5, 3, 224, 224)
    model = BiSalNet()
    am, a1, a2, a3 = model(x)
    print(am.shape)
    print(a1.shape)
    print(a2.shape)
    print(a3.shape)
