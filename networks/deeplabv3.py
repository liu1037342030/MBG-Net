import torch
import torch.nn as nn
import torch.nn.functional as F
# from networks.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from torch.nn import SyncBatchNorm as SynchronizedBatchNorm2d
from networks.aspp import build_aspp
from networks.decoder import build_decoder
from networks.backbone import build_backbone


class DeepLab(nn.Module):

    def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                 sync_bn=False, freeze_bn=False):
        super(DeepLab, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)

        if freeze_bn:
            self.freeze_bn()

    def forward(self, input):
        #backbone  x送入aspp  low_level_feat为backbone的分�?        
        x, low_level_feat = self.backbone(input)
        #print('11111111111')
        #print(x.shape)
        #print(low_level_feat.shape)
        #得到的为4xup的特征图
        feature = self.aspp(x)
        #print('2222222')
        #print(feature.shape)

        #在这加transformer  输入的是low_level_feat  输出的上采样后和aspp出来的进行cat操作
        #问题难点  如何对输入的图片进行处理 送入transformer�?输出后的结果如何进行上采�?        #需要参考相关代码看看如何操�?        #把经过transformer出来的x代替low_level_feat送入decoder

        x1, x2 = self.decoder(feature, low_level_feat)
        #print('333333')

        x2 = F.interpolate(x2, size=input.size()[2:], mode='bilinear', align_corners=True)
        x1 = F.interpolate(x1, size=input.size()[2:], mode='bilinear', align_corners=True)

        #print(x1.shape)
        #print(x2.shape)

        # return x1, x2, feature
        return x1, x2

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p