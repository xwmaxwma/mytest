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
    def __init__(self, ignore_index=255, num_classes=6):
        super().__init__()
        self.ignore_index = ignore_index
        self.num_classes = num_classes
    
    def forward(self, x, gt):
        B, C, H, W = x.size()
        label = F.interpolate(gt.float(), size=(H, W), mode='nearest')
        label = label.squeeze(1)    # (B, H, W)
        not_ignore_spatial_mask = (label.to(torch.int32) != self.ignore_index)  # (B, H, W)
        label = label * not_ignore_spatial_mask
        one_hot_label = F.one_hot(label.to(torch.long), self.num_classes) # (B, H, W, K)
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
    def __init__(self, channel=512, num_classes=6, patch_size=(2,2), momentum=0.9, num_prototype_per_class=8, ignore_index=255, out_scale=10):
        super(ECC, self).__init__()
        self.momentum = momentum
        self.num_prototype_per_class = num_prototype_per_class
        self.channel = channel
        self.num_classes = num_classes
        self.prototype = nn.Parameter(torch.randn(self.num_classes, self.num_prototype_per_class, self.channel),requires_grad=False) #(k, m, c)
        
        self.patch_split = Patch_Split(patch_size)
        self.gt2center = GT2Center(ignore_index=ignore_index, num_classes=num_classes)
        
        self.out_scale = out_scale
        
        self.gumbel_softmax = gumbel_softmax
    
    # center: (B', k, c), B'=B*num_h*num_w
    def momentum_update(self, center):
        bs, k, c = center.size()
        center = center.permute(1, 0, 2) #(k, B', c)
        proto_center_dist = torch.matmul(center, self.prototype.permute(0, 2, 1)) #(k, B', m)
        
        nearest_indices = self.gumbel_softmax(proto_center_dist, dim=-1, hard=True, tau=1.0) #(k, B', m)
        nearest_indices = nearest_indices.permute(0, 2, 1) #(k, m, B')
        
        # nearest_indices = torch.argmin(proto_center_dist, dim=-1) #(k, B')  # 这里应该是argmax！！所以之前的结果很低
        # nearest_indices = F.one_hot(nearest_indices.long(), proto_center_dist.size(-1)).float().permute(0, 2, 1) #(k, m, B')
        new_proto = torch.matmul(nearest_indices, center) #(k, m, c)
        
        non_zero_map = torch.count_nonzero(nearest_indices, dim=-1) #(k, m)
        # non_zero_map = torch.where(non_zero_map == 0, 
        #                             torch.tensor([1], device=non_zero_map.device, dtype=torch.long),
        #                             non_zero_map) #(k, m)
        # new_proto = new_proto / non_zero_map.unsqueeze(-1).repeat(1, 1, c)
        
        # self.prototype.data = self.momentum * self.prototype.data + (1 - self.momentum) * new_proto
        
        #non_zero_map = non_zero_map + 1
        # new_proto = (new_proto + self.prototype.data) / non_zero_map.unsqueeze(-1).repeat(1, 1, c)
        # self.prototype.data = new_proto
        new_proto = new_proto / non_zero_map.unsqueeze(-1).repeat(1, 1, c) * self.momentum + self.prototype.data * (1-self.momentum)
        self.prototype.data = new_proto
        
    # x: (B, C, H, W), gt: (B, H, W)
    def forward(self, x, gt=None):
        B, C, H, W = x.size()
        if self.training:
            patch_x, patch_gt = self.patch_split(x), self.patch_split(gt)
            center = self.gt2center(patch_x, patch_gt)
            self.momentum_update(center)
        proto = self.prototype.reshape(-1, C) #(k*m, c)
        # x1 = x.permute(0, 2, 3, 1).reshape(-1, C) #(B*H*W, C)
        
        # x1, proto1 = F.normalize(x1, 2, 1), F.normalize(proto, 2, 1)
        # pixel_proto_dist1 = torch.matmul(x1, proto1.permute(1, 0))# (B*H*W, k*m)
        x, proto = F.normalize(x, 2, 1), F.normalize(proto, 2, 1)
        proto = proto.unsqueeze(-1).unsqueeze(-1) #(k*m, c, 1, 1)
        pixel_proto_dist = F.conv2d(x, weight=proto)
        
        pixel_proto_dist = pixel_proto_dist.view(B*H*W, self.num_classes, -1).permute(1, 0, 2) #(k, B*H*W, m)
        proto_class_dist = torch.max(pixel_proto_dist, dim=-1)[0] #(k, B*H*W)
        proto_class_dist = proto_class_dist.view(self.num_classes, B, H, W).permute(1, 0, 2, 3) #(B, K, H, W)

        return proto_class_dist * self.out_scale

        
def hard_softmax(logits, dim):
    y_soft = logits.softmax(dim)
    # Straight through.
    index = y_soft.max(dim, keepdim=True)[1]
    y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
    ret = y_hard - y_soft.detach() + y_soft

    return ret

def gumbel_softmax(logits: torch.Tensor, tau: float = 1, hard: bool = True, dim: int = -1) -> torch.Tensor:

    gumbel_dist = torch.distributions.gumbel.Gumbel(
        torch.tensor(0., device=logits.device, dtype=logits.dtype),
        torch.tensor(1., device=logits.device, dtype=logits.dtype))
    gumbels = gumbel_dist.sample(logits.shape)

    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret

if __name__ == "__main__":
    x = torch.randn(4, 512, 64, 64)
    gt = torch.randint(0, 6, (4, 1, 64, 64))
    net = ECC()
    res = net(x, gt)
    res = F.softmax(res, dim=1)
    res = hard_softmax(res, dim=1)
    print(res.shape)
    
