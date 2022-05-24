from networks.backbone import resnet, xception, drn, mobilenet, mobilenetv3


def build_backbone(backbone, output_stride, bn):
    if backbone == 'resnet':
        return resnet.ResNet50(output_stride, bn)
    elif backbone == 'xception':
        return xception.AlignedXception(output_stride, bn)
    elif backbone == 'drn':
        return drn.drn_d_54(bn)
    elif backbone == 'mobilenet':
        return mobilenet.MobileNetV2(output_stride, bn)
    elif backbone == 'mobilenetv3':
        return mobilenetv3.mobilenet_v3_large()
    else:
        raise NotImplementedError