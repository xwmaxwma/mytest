"""
Name: ECC-Explicit Class Classifier
Author: Xiaowen Ma
Exp: v1
"""

import torch
from torch import nn 
from torch.nn import functional as F

# from .ocr_head import SpatialGatherModule

class Patch_Split(nn.Module):
    """Split feature map to patch
    
    Args:
        patch_size (num_h, num_w): the number of patches
    Shape:
        x: (B, C, H, W)
        out: (B*num_h*num_w, C, h, w)
    
    """
    def __init__(self, patch_size):
        super(Patch_Split, self).__init__()
        self.patch_size = patch_size
    
    def forward(self, x):
        B, C, H, W = x.size()
        num_h, num_w = self.patch_size
        patch_h, patch_w = H // num_h, W // num_w
        out = x.view(B, C, num_h, patch_h, num_w, patch_w)
        out = out.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, patch_h, patch_w)
        return out
        
class GT2Center(nn.Module):
    """Obtain class center with gt label
    
    Shape:
        feat: (B, C, H, W), gt: (B, 1, H, W)
        out: (B, K, C)
    """
    def __init__(self, ignore_index=-1, num_classes=6):
        super().__init__()
        self.ignore_index = ignore_index
        self.num_classes = num_classes
    
    def forward(self, x, gt):
        B, C, H, W = x.size()
        label = F.interpolate(gt.float(), size=(H, W), mode='nearest')
        label = label.squeeze(1)    # (B, H, W)
        not_ignore_spatial_mask = label.to(torch.int32) != self.ignore_index  # (B, H, W)
        one_hot_label = F.one_hot((label * not_ignore_spatial_mask).to(torch.long), self.num_classes) # (B, H, W, K)
        one_hot_label = (one_hot_label.permute(3, 0, 1, 2) * not_ignore_spatial_mask).float().permute(1, 0, 2, 3) # (B, K, H, W)
        
        one_hot_label_flat = one_hot_label.view(B, self.num_classes, -1) #(B, k, H*W)
        non_zero_map = torch.count_nonzero(one_hot_label_flat, dim=-1)
        non_zero_map = torch.where(non_zero_map == 0, 
                                    torch.tensor([1], device=non_zero_map.device, dtype=torch.long),
                                    non_zero_map) #(B, K)

        center = torch.matmul(one_hot_label_flat, x.view(B, C, -1).permute(0, 2, 1)) # (B, K, C)
        center = center / non_zero_map.unsqueeze(-1).repeat(1, 1, C)
        
        return center

class ECC(nn.Module):
    def __init__(self, in_channel=512, out_channel=512, num_classes=6, patch_size=(4,4), momentum=0.5, num_prototype_per_class=8, ignore_index=6):
        super(ECC, self).__init__()
        self.momentum = momentum
        self.num_prototype_per_class = num_prototype_per_class
        self.out_channel = out_channel
        self.num_classes = num_classes
        self.prototype = nn.Parameter(torch.randn(self.num_classes, self.num_prototype_per_class, self.out_channel),requires_grad=True) #(k, m, c)
        
        self.patch_split = Patch_Split(patch_size)
        self.gt2center = GT2Center(ignore_index=ignore_index, num_classes=num_classes)
    
    # center: (B', k, c), B'=B*num_h*num_w
    def momentum_update(self, center):
        bs, k, c = center.size()
        center = center.permute(1, 0, 2) #(k, B', c)
        proto_center_dist = torch.cdist(center, self.prototype) # (k, B', m)
        nearest_indices = torch.argmin(proto_center_dist, dim=-1)
        for k_ in range(k):
            for bs_ in range(bs):
                self.prototype.data[k_, nearest_indices[k_][bs_], :] = self.momentum * self.prototype.data[k_, nearest_indices[k_][bs_], :] + (1 - self.momentum) * center[k_][bs_]
    # x: (B, C, H, W), gt: (B, H, W)
    def forward(self, x, gt):
        gt = gt.unsqueeze(1)
        B, C, H, W = x.size()
        patch_x, patch_gt = self.patch_split(x), self.patch_split(gt)
        if self.training:
            center = self.gt2center(patch_x, patch_gt)
            self.momentum_update(center)
        proto = self.prototype.reshape(-1, C) #(k*m, c)
        x = x.permute(0, 2, 3, 1).reshape(-1, C) #(B*H*W, C)
        pixel_proto_dist = torch.cdist(x, proto) # (B*H*W, k*m)
        pixel_proto_dist = pixel_proto_dist.view(B*H*W, self.num_classes, -1).permute(1, 0, 2) #(k, B*H*W, m)
        proto_class_dist = torch.max(pixel_proto_dist, dim=-1)[0] #(k, B*H*W)
        proto_class_dist = proto_class_dist.view(self.num_classes, B, H, W).permute(1, 0, 2, 3) #(B, K, H, W)
        return proto_class_dist
        
        

if __name__ == "__main__":
    x = torch.randn(4, 512, 64, 64)
    gt = torch.randint(0, 6, (4, 64, 64))
    net = ECC()
    res = net(x, gt)
    print(res.shape)
    