import numpy as np
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
#from Models.fast_scnn_decoder_SEM import *

class BiSalNet(nn.Module):
    def __init__(self):
        super(BiSalNet, self).__init__()
        self.context_path = FastSCNN()
        self.ARMs = nn.ModuleList([
                AttentionRefinement(in_channel=128, out_channel=128),
                AttentionRefinement(in_channel=64, out_channel=64),
                convbnrelu(64, 64, 1, 1, 0),#AttentionRefinement(in_channel=64, out_channel=64),
                convbnrelu(48, 48, 1, 1, 0),#AttentionRefinement(in_channel=48, out_channel=48),
                convbnrelu(32, 16, 1, 1, 0)#AttentionRefinement(in_channel=32, out_channel=32)
                ])
        self.refines = nn.ModuleList([
                DSConv3x3(128, 64, stride=1),
                DSConv3x3(64, 64, stride=1),
                DSConv3x3(64, 48, stride=1),
                DSConv3x3(48, 16, stride=1),
                DSConv3x3(32, 16, stride=1)
                ])
        self.heads = nn.ModuleList([
                SalHead(in_channel=64, middle_channel=64),
                SalHead(in_channel=64, middle_channel=64),
                SalHead(in_channel=48, middle_channel=48),
                SalHead(in_channel=16, middle_channel=16),
                SalHead(in_channel=16, middle_channel=16)
                ])
    
    def forward(self, x): # (3, 1)
        ct_stage1, ct_stage2, ct_stage3, ct_stage4, ct_stage5, ct_stage6 = self.context_path(x)
        # (32, 1/2) (48, 1/4) (64, 1/8)  (64, 1/16) (128, 1/32) (128, 1/32)
        
        fused_stage1 = interpolate(self.ARMs[0](ct_stage5) + ct_stage6, ct_stage4.size()[2:]) # (128, 1/16)
        refined1 = self.refines[0](fused_stage1) # (64, 1/16)
        
        fused_stage2 = interpolate(self.ARMs[1](ct_stage4) + refined1, ct_stage3.size()[2:]) # (64, 1/8)
        refined2 = self.refines[1](fused_stage2) # (64, 1/8)
        
        fused_stage3 = interpolate(self.ARMs[2](ct_stage3) + refined2, ct_stage2.size()[2:]) # (64, 1/4)
        refined3 = self.refines[2](fused_stage3) # (48, 1/4)
        
        fused_stage4 = interpolate(self.ARMs[3](ct_stage2) + refined3, ct_stage1.size()[2:]) # (48, 1/2)
        refined4 = self.refines[3](fused_stage4) # (32, 1/2)
        
        #fused_stage5 = interpolate(self.ARMs[4](ct_stage1) + refined4, x.size()[2:])         # (32, 1)
        #refined5 = self.refines[4](fused_stage5) # (16, 1)
        refined5 = self.ARMs[4](ct_stage1) + refined4

        output_side1 = interpolate(self.heads[0](refined1), x.size()[2:])
        output_side2 = interpolate(self.heads[1](refined2), x.size()[2:])
        output_side3 = interpolate(self.heads[2](refined3), x.size()[2:])
        output_side4 = interpolate(self.heads[3](refined4), x.size()[2:])
        output_main  = interpolate(self.heads[4](refined5), x.size()[2:])
        return output_main, output_side1, output_side2, output_side3, output_side4

interpolate = lambda x, size: F.interpolate(x, size=size, mode="bilinear", align_corners=True)

class convbnrelu(nn.Module):
    def __init__(self, in_channel, out_channel, k=3, s=1, p=1, g=1, d=1, bias=False, bn=True, relu=True):
        super(convbnrelu, self).__init__()
        conv = [nn.Conv2d(in_channel, out_channel, k, s, p, dilation=d, groups=g, bias=bias)]
        if bn:
            conv.append(nn.BatchNorm2d(out_channel))
        if relu:
            conv.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*conv)
        
    def forward(self, x):
        return self.conv(x)

class DSConv3x3(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, dilation=1, relu=True):
        super(DSConv3x3, self).__init__()
        self.conv = nn.Sequential(
                convbnrelu(in_channel, in_channel, k=3, s=stride, p=dilation, d=dilation, g=in_channel),
                convbnrelu(in_channel, out_channel, k=1, s=1, p=0, relu=relu)
                )
    def forward(self, x):
        return self.conv(x)
    
class AttentionRefinement(nn.Module):
    def __init__(self, in_channel, out_channel=128):
        super(AttentionRefinement, self).__init__()
        self.conv3x3 = DSConv3x3(in_channel, out_channel, stride=1)
        self.SE_Attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                convbnrelu(out_channel, out_channel, 1, 1, 0, relu=False),
                nn.Sigmoid()
                )
    
    def forward(self, x):
        conv = self.conv3x3(x)
        conv_se = self.SE_Attention(conv)
        return conv * conv_se
        
class SalHead(nn.Module):
    def __init__(self, in_channel, middle_channel):
        super(SalHead, self).__init__()
        self.conv = nn.Sequential(
                DSConv3x3(in_channel, middle_channel, stride=1),
                nn.Conv2d(middle_channel, 1, 1, stride=1, padding=0),
                nn.Sigmoid()
                )
    
    def forward(self, x):
        return self.conv(x)

class FastSCNN(nn.Module):
    def __init__(self):
        super(FastSCNN, self).__init__()
        self.layer1 = convbnrelu(3, 32, k=3, s=2, p=1)
        self.layer2 = nn.Sequential(
                DSConv3x3(32, 48, stride=2),
                ScaleEnhance(48, 48)
                )
        self.layer3 = nn.Sequential(
                DSConv3x3(48, 64, stride=2),
                ScaleEnhance(64, 64)
                )
        self.layer4 = nn.Sequential(
                LinearBottleneck(64, 64, expansion=6, stride=2),
                LinearBottleneck(64, 64, expansion=6, stride=1),
                LinearBottleneck(64, 64, expansion=6, stride=1)
                )
        self.layer5 = nn.Sequential(
                LinearBottleneck(64, 96, expansion=6, stride=2),
                LinearBottleneck(96, 96, expansion=6, stride=1),
                LinearBottleneck(96, 96, expansion=6, stride=1),
                LinearBottleneck(96, 128, expansion=6, stride=1),
                LinearBottleneck(128, 128, expansion=6, stride=1),
                LinearBottleneck(128, 128, expansion=6, stride=1)
                )
        self.layer6 = PyramidPooling(128, 128)

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out6 = self.layer6(out5)
        return out1, out2, out3, out4, out5, out6

class ScaleEnhance(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ScaleEnhance, self).__init__()
        self.conv = DSConv3x3(in_channel, in_channel, stride=1)
        self.branches = nn.ModuleList([
                DSConv3x3(in_channel, in_channel, stride=1, dilation=1),
                DSConv3x3(in_channel, in_channel, stride=1, dilation=2),
                DSConv3x3(in_channel, in_channel, stride=1, dilation=4),
                DSConv3x3(in_channel, in_channel, stride=1, dilation=8)                
                ])
        self.fuse = convbnrelu(in_channel, out_channel, k=1, s=1, p=0, relu=False)
    
    def forward(self, x):
        x = self.conv(x)
        br1, br2, br3, br4 = [branch(x) for branch in self.branches]
        return self.fuse(x + br1 + br2 + br3 + br4)

class LinearBottleneck(nn.Module):
    def __init__(self, in_channel, out_channel, expansion=6, stride=1):
        super(LinearBottleneck, self).__init__()
        self.use_shortcut = (stride == 1 and in_channel == out_channel)
        self.block = nn.Sequential(
                convbnrelu(in_channel, in_channel*expansion, k=1, s=1, p=0),
                DSConv3x3(in_channel*expansion, out_channel, stride=stride, relu=False)
                )

    def forward(self, x):
        out = self.block(x)
        if self.use_shortcut:
            out = x + out
        return out

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
        
def computeTime(model, device='cuda'):
    inputs = torch.randn(100, 3, 224, 224)
    if device == 'cuda':
        model = model.cuda()
        inputs = inputs.cuda()

    model.eval()

    time_spent = []
    for idx in range(100):
        start_time = time.time()
        with torch.no_grad():
            _ = model(inputs)

        if device == 'cuda':
            torch.cuda.synchronize()  # wait for cuda to finish (cuda is asynchronous!)
        if idx > 10:
            time_spent.append(time.time() - start_time)
    print('Avg execution time (ms): {:.4f}'.format(np.mean(time_spent)))

torch.backends.cudnn.benchmark = True

model = BiSalNet().cuda()
inputs = torch.randn(100, 3, 224, 224).cuda()
time_record = []

for idx in range(100):
    start_time = time.time()
    with torch.no_grad():
        _ = model(inputs)
    torch.cuda.synchronize()  # wait for cuda to finish (cuda is asynchronous!)

    time_record.append(time.time() - start_time)

print('Avg Execution time (ms): {:.4f} (std: {:.4f})'.format(np.mean(time_record[10:]), np.std(time_record[10:])))