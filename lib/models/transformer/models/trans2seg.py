import torch
import torch.nn as nn
import torch.nn.functional as F

from .segbase import SegBaseModel
from .model_zoo import MODEL_REGISTRY
from ..modules import _ConvBNReLU, SeparableConv2d, _ASPP, _FCNHead
from ..config import cfg
from ..modules import VisionTransformer
from IPython import embed


__all__ = ['Trans2Seg']


@MODEL_REGISTRY.register(name='Trans2Seg')
class Trans2Seg(nn.Module):

    def __init__(self):
        super().__init__()
        c1_channels = 256
        c4_channels = 2048

        vit_params = cfg.MODEL.TRANS2Seg
        hid_dim = 64

        c4_HxW = 48

        vit_params['decoder_feat_HxW'] = c4_HxW

        self.transformer_head = TransformerHead(vit_params, c1_channels=c1_channels, c4_channels=c4_channels, hid_dim=hid_dim)
        self.__setattr__('decoder', ['transformer_head'])


    def forward(self, c1,c4):
        attn = self.transformer_head(c4, c1)
        return attn


class Transformer(nn.Module):
    def __init__(self, c4_channels=2048):
        super().__init__()
        #last_channels = vit_params['embed_dim']
        self.vit = VisionTransformer(input_dim=c4_channels,
                                     embed_dim=256,
                                     depth=8,
                                     num_heads=8,
                                     mlp_ratio=3.,
                                     decoder_feat_HxW=48)

    def forward(self, x):
        n, _, h, w = x.shape
        x = self.vit.hybrid_embed(x)

        cls_token, x = self.vit.forward_encoder(x)

        attns_list = self.vit.forward_decoder(x)

        x = x.reshape(n, h, w, -1).permute(0, 3, 1, 2)
        return x, attns_list


class TransformerHead(nn.Module):
    def __init__(self, vit_params, c1_channels=256, c4_channels=2048, hid_dim=64, norm_layer=nn.BatchNorm2d):
        super().__init__()

        last_channels = 256
        nhead = 8

        self.transformer = Transformer(c4_channels=c4_channels)

        self.conv_c1 = _ConvBNReLU(c1_channels, hid_dim, 1, norm_layer=norm_layer)

        self.lay1 = SeparableConv2d(last_channels+nhead, hid_dim, 3, norm_layer=norm_layer, relu_first=False)
        self.lay2 = SeparableConv2d(hid_dim, hid_dim, 3, norm_layer=norm_layer, relu_first=False)
        self.lay3 = SeparableConv2d(hid_dim, hid_dim, 3, norm_layer=norm_layer, relu_first=False)

        self.pred = nn.Conv2d(hid_dim, 1, 1)
        self.pred2 = nn.Conv2d(nhead, 1, 1)

        #self.deconv_layer = self._make_deconv_layer(3, [hid_dim, hid_dim, hid_dim], [4,4,4])
        
    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding
        
        
    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            self.inplanes=64
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False))
            layers.append(nn.BatchNorm2d(planes, momentum=0.9))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)
        

    def forward(self, x, c1):
        feat_enc, attns_list = self.transformer(x)
        attn_map = attns_list[-1]
        B, nclass, nhead, _ = attn_map.shape
        _, _, H, W = feat_enc.shape
        attn_map = attn_map.reshape(B*nclass, nhead, H, W)

        x = torch.cat([_expand(feat_enc, nclass), attn_map], 1)

        x = self.lay1(x)
        x = self.lay2(x)
        size = c1.size()[2:]
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        #x = self.deconv_layer(x)
        c1 = self.conv_c1(c1)
        x = x + _expand(c1, nclass)

        x = self.lay3(x)
        x = self.pred(x).reshape(B, nclass, size[0], size[1])
        #attn = self.pred2(attn_map).reshape(B, nclass, size[0], size[1])

        return x#, attn

def _expand(x, nclass):
    return x.unsqueeze(1).repeat(1, nclass, 1, 1, 1).flatten(0, 1)
