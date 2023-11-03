# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from .psp_head import PPM
from mmcv.runner import force_fp32
import os
import math
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class COBlock(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, drop_out=0.):
        super(COBlock, self).__init__()
        inner_dim = heads * dim_head
        
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.drop_out = nn.Dropout(drop_out)
        self.soft_hw = nn.Softmax(dim=-2)
        self.soft_k = nn.Softmax(dim=-1)
        
        self.q_proj = nn.Linear(dim, inner_dim, bias=False)
        self.k_proj = nn.Linear(dim, inner_dim, bias=False)
        self.v1_proj = nn.Linear(dim, inner_dim, bias=False)
        self.v2_proj = nn.Linear(dim, inner_dim, bias=False)
        self.out1_proj = nn.Linear(inner_dim, dim, bias=False)
        self.out2_proj = nn.Linear(inner_dim, dim, bias=False)
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        # self.norm3 = nn.LayerNorm(dim)
        # self.norm4 = nn.LayerNorm(dim)
        
        self.ffn1_proj = nn.Sequential(
            nn.Linear(dim, dim, bias=False), 
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
        )   
        self.ffn2_proj = nn.Sequential(
            nn.Linear(dim, dim, bias=False), 
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
        ) 
        

        
    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos
    
    def forward(self, feat, center, pos, center_pos):
        # print(center.shape, center_pos.shape)
        pos = pos.flatten(2).permute(0, 2, 1)
        query_feat = self.with_pos_embed(feat, pos)
        key_feat = self.with_pos_embed(center, center_pos)
        q = self.q_proj(query_feat) # (B, HW, C)
        k = self.k_proj(key_feat) # (B, K, C)
        v1 = self.v1_proj(feat)
        v2 = self.v2_proj(center)
        
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads) #(b, h, HW, c)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads) #(b, h, k, c)
        v1 = rearrange(v1, 'b n (h d) -> b h n d', h=self.heads) #(b, h, HW, c)
        v2 = rearrange(v2, 'b n (h d) -> b h n d', h=self.heads) #(b, h, k, c)
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale #(b, h, HW, k)
        
        attn1, attn2 = self.soft_hw(attn), self.soft_k(attn)
        attn1, attn2 = self.drop_out(attn1), self.drop_out(attn2)

        out_feat = torch.matmul(attn2, v2) #(b, h, HW, c)
        out_center = torch.matmul(attn1.transpose(-1, -2), v1) #(b, h, k, c)
        
        out_feat = rearrange(out_feat, 'b h n d -> b n (h d)') #(b, HW, C)
        out_center = rearrange(out_center, 'b h n d -> b n (h d)') #(b, k, C)
        
        out_feat = self.out1_proj(out_feat) + feat
        out_center = self.out2_proj(out_center) + center
        
        out_feat, out_center = self.norm1(out_feat), self.norm2(out_center)
        out_feat, out_center = self.ffn1_proj(out_feat) + out_feat, self.ffn2_proj(out_center) + out_center
        
        return out_feat, out_center
               

@HEADS.register_module()
class CONetV5(BaseDecodeHead):
    def __init__(self, pool_scales=(1, 2, 3, 6), **kwargs):
        super(CONetV5, self).__init__(
            input_transform='multiple_select', **kwargs)
        # PSP Module
        self.psp_modules = PPM(
            pool_scales,
            self.in_channels[-1],
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)
        self.bottleneck = ConvModule(
            self.in_channels[-1] + len(pool_scales) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = ConvModule(
                in_channels,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = ConvModule(
            len(self.in_channels) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        # self.proj = nn.Sequential(
        #     nn.Linear(self.channels*2, self.channels//2, bias=False), 
        #     nn.ReLU(inplace=True),
        #     nn.Linear(self.channels//2, self.channels),
        # )                 
        self.apd_proj = nn.Sequential(
            nn.Linear(self.channels*2, self.channels//2, bias=False), 
            nn.ReLU(inplace=True),
            nn.Linear(self.channels//2, self.channels),
        )       
        self.conv_seg = nn.Conv2d(self.channels, self.num_classes, kernel_size=1)      
        self.dropout = nn.Dropout2d(self.dropout_ratio)         
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)         
        self.com = COBlock(self.channels)
        self.pos = PositionEmbeddingSine(self.channels//2, normalize=True)
        
        
    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)

        return output

    def forward(self, inputs):
        """Forward function."""

        inputs = self._transform_inputs(inputs)

        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        laterals.append(self.psp_forward(inputs))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + resize(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)

        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = resize(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        fpn_outs = torch.cat(fpn_outs, dim=1)
        feat = self.fpn_bottleneck(fpn_outs)
        out = self.conv_seg(self.dropout(feat))
        return out, feat

    def get_pred(self, x, proto):
        b, c, h, w = x.size()[:]
        # x: [b, c, h, w]
        # proto: [b, cls, c]  
        cls_num = proto.size(1)
        x = x / (torch.norm(x, 2, 1, True) + 1e-12)
        proto = proto / (torch.norm(proto, 2, -1, True) + 1e-12) # b, n, c
        x = x.contiguous().view(b, c, h*w)  # b, c, hw
        pred = proto @ x  # b, cls, hw
        pred = pred.contiguous().view(b, cls_num, h, w)
        return pred * 15

    def get_adaptive_perspective(self, x, y, new_proto, proto):
        raw_x = x.clone()
        # y: [b, h, w]
        # x: [b, c, h, w]
        b, c, h, w = x.shape[:]
        if y.dim() == 3:
            y = y.unsqueeze(1)        
        y = F.interpolate(y.float(), size=(h, w), mode='nearest')  # b, 1, h, w
        unique_y = list(y.unique())
        if 255 in unique_y:
            unique_y.remove(255)

        tobe_align = []
        label_list = []
        for tmp_y in unique_y:
            tmp_mask = (y == tmp_y).float()
            tmp_proto = (x * tmp_mask).sum(0).sum(-1).sum(-1) / (tmp_mask.sum(0).sum(-1).sum(-1) + 1e-12)
            onehot_vec = torch.zeros(new_proto.shape[0], 1).cuda()  # cls, 1
            onehot_vec[tmp_y.long()] = 1
            new_proto = new_proto * (1 - onehot_vec) + tmp_proto.unsqueeze(0) * onehot_vec
            tobe_align.append(tmp_proto.unsqueeze(0))
            label_list.append(tmp_y)  
        
        new_proto = torch.cat([new_proto, proto], -1)
        new_proto = self.apd_proj(new_proto) #(k, c)
        new_proto = new_proto.unsqueeze(-1).unsqueeze(-1)   # cls, 512, 1, 1
        new_proto = F.normalize(new_proto, 2, 1)
        raw_x = F.normalize(raw_x, 2, 1)
        pred = F.conv2d(raw_x, weight=new_proto) * 15
        return pred

    # x(B, C, H, W) proto(B, K, C)
    def feat_proto(self, x, proto, pos, proto_pos):
        B, C, H, W = x.size()
        x = x.flatten(2).permute(0, 2, 1) # (B, HW, C)
        new_x, new_proto = self.com(x, proto, pos, proto_pos)
        new_x = new_x.reshape(B, H, W, C).permute(0, 3, 1, 2) #(B, C, H, W)
        return new_x, new_proto
        
    
    def post_refine_proto_v2(self, x, pred, proto):    
        pos = self.pos(x)
        raw_x = x.clone()        
        b, c, h, w = raw_x.shape[:]
        pred = pred.view(b, proto.shape[0], h*w)
        pred = F.softmax(pred, 1)   # b, k, hw
        proto_pos = (pred @ raw_x.view(b, c, h*w).permute(0, 2, 1)) / (pred.sum(-1).unsqueeze(-1) + 1e-12) # (b, k, c)

        new_feat, new_proto = self.feat_proto(x, proto.unsqueeze(0).repeat(b, 1, 1), pos, proto_pos)
        
        new_pred = self.get_pred(new_feat, new_proto)
        return new_pred, new_feat

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        y = gt_semantic_seg
        pred_first, feat = self.forward(inputs)

        pre_self_x = pred_first.clone()
        pred_pred, new_feat = self.post_refine_proto_v2(x=feat, pred=pred_first, proto=self.conv_seg.weight.squeeze())      
        pred_gt = self.get_adaptive_perspective(x=new_feat, y=y, new_proto=self.conv_seg.weight.detach().data.squeeze(), proto=self.conv_seg.weight.squeeze())   

        kl_loss = get_distill_loss(pred=pred_pred, soft=pred_gt.detach(), target=y.squeeze(1))

        pre_self_x = F.interpolate(pre_self_x, size=y.shape[-2:], mode='bilinear', align_corners=True)
        pre_self_loss = self.criterion(pre_self_x, y.squeeze(1).long()) 
        pred_gt = F.interpolate(pred_gt, size=y.shape[-2:], mode='bilinear', align_corners=True)
        pred_gt_loss = self.criterion(pred_gt, y.squeeze(1).long()) 

        losses = self.losses(pred_pred, y)
        losses['firstLoss'] =  pre_self_loss.detach().data
        losses['gtLoss'] =  pred_gt_loss.detach().data
        losses['KLLoss'] =  kl_loss.detach().data
        losses['predLoss'] =  losses['loss_ce'].detach().data
        losses['loss_ce'] = losses['loss_ce'] + pre_self_loss + pred_gt_loss + kl_loss
        return losses      

    def forward_test(self, inputs, img_metas, test_cfg):
        x, feat = self.forward(inputs)
        x, new_feat = self.post_refine_proto_v2(x=feat, pred=x, proto=self.conv_seg.weight.squeeze())      
        return x    


def get_distill_loss(pred, soft, target, smoothness=0.5, eps=0):
    '''
    knowledge distillation loss
    '''
    b, c, h, w = soft.shape[:]
    soft.detach()
    target = F.interpolate(target.unsqueeze(1).float(), size=pred.shape[-2:], mode='nearest').squeeze(1).long()
    onehot = target.view(-1, 1) # bhw, 1
    ignore_mask = (onehot == 255).float()
    onehot = onehot * (1 - ignore_mask) 
    onehot = torch.zeros(b*h*w, c).cuda().scatter_(1,onehot.long(),1)  # bhw, n
    onehot = onehot.contiguous().view(b, h, w, c).permute(0, 3, 1, 2)   # b, n, h, w
    sm_soft = F.softmax(soft / 1, 1)
    smoothed_label = smoothness * sm_soft + (1 - smoothness) * onehot
    if eps > 0: 
        smoothed_label = smoothed_label * (1 - eps) + (1 - smoothed_label) * eps / (smoothed_label.shape[1] - 1) 

    loss = torch.mul(-1 * F.log_softmax(pred, dim=1), smoothed_label)   # b, n, h, w
    
    sm_soft = F.softmax(soft / 1, 1)   # b, c, h, w    
    entropy_mask = -1 * (sm_soft * torch.log(sm_soft + 1e-12)).sum(1)
    loss = loss.sum(1) 

    ### for class-wise entropy estimation    
    unique_classes = list(target.unique())
    if 255 in unique_classes:
        unique_classes.remove(255)
    valid_mask = (target != 255).float()
    entropy_mask = entropy_mask * valid_mask
    loss_list = []
    weight_list = []
    for tmp_y in unique_classes:
        tmp_mask = (target == tmp_y).float()
        tmp_entropy_mask = entropy_mask * tmp_mask
        class_weight = 1
        tmp_loss = (loss * tmp_entropy_mask).sum() / (tmp_entropy_mask.sum() + 1e-12)
        loss_list.append(class_weight * tmp_loss)
        weight_list.append(class_weight)
    if len(weight_list) > 0:
        loss = sum(loss_list) / (sum(weight_list) + 1e-12)
    else:
        loss = torch.zeros(1).cuda().mean()
    return loss

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos
    
    def __repr__(self, _repr_indent=4):
        head = "Positional encoding " + self.__class__.__name__
        body = [
            "num_pos_feats: {}".format(self.num_pos_feats),
            "temperature: {}".format(self.temperature),
            "normalize: {}".format(self.normalize),
            "scale: {}".format(self.scale),
        ]
        # _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)

# dual class optimization
# 重写conet，将特征和中心都构造成位置查询和内容查询的方式
