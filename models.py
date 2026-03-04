"""
models.py
---------
CCoMAML network architectures.

BaseNet  — Meta-learner with dual ResNet-50 / ViT backbone and
           Multi-Head Attention Feature Fusion (MHAFF).
           Inner-loop adaptable parameters: res_linear, trans_linear, fc.

CoLearner — Auxiliary network f_psi that estimates the gradient
            correction Delta_g_i from the previous adapted parameters
            phi_{i-1}' and the query images D_i^Qu.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torchvision import models
from transformers import ViTModel

from .meta_modules import MetaModule, MetaLinear, get_subdict


class BaseNet(MetaModule):
    """Dual-backbone meta-learner (MHAFF).

    Backbone layout
    ---------------
    ResNet-50  : layers 0–6 frozen; layer3 + layer4 trainable (outer loop).
    ViT-B/16   : fully frozen (feature extractor only).

    Adaptable parameters (inner loop — theta)
    -----------------------------------------
    res_linear   : 2048 -> 64  projection for ResNet features.
    trans_linear : 768  -> 64  projection for ViT features.
    fc           : 64   -> num_classes  classification head.

    Fusion
    ------
    Multi-head self-attention over the two 64-d feature sequences.

    Args:
        num_classes (int): Number of ways (N-way classification head size).
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()

        # --- ResNet-50 backbone (layer3, layer4 unfrozen) ---
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        for name, param in self.resnet.named_parameters():
            if not (name.startswith('layer3') or name.startswith('layer4')):
                param.requires_grad = False

        # --- ViT-B/16 backbone (fully frozen) ---
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        for param in self.vit.parameters():
            param.requires_grad = False

        # --- Inner-loop adaptable projections (MetaLinear = theta) ---
        self.res_linear   = MetaLinear(2048, 64)
        self.trans_linear = MetaLinear(768,  64)
        self.fc           = MetaLinear(64,   num_classes)

        # --- Attention fusion (outer-loop only, standard nn) ---
        self.self_attention = nn.MultiheadAttention(
            embed_dim=64, num_heads=8, batch_first=False
        )

    def forward(self, trans_b: torch.Tensor, res_b: torch.Tensor,
                params: OrderedDict = None):
        """Forward pass with optional adapted parameter injection.

        Args:
            trans_b (Tensor): ViT-preprocessed images   (N, 3, 224, 224).
            res_b   (Tensor): ResNet-preprocessed images (N, 3, 224, 224).
            params  (OrderedDict | None): Adapted phi' from inner loop.
                If None, the module's own registered theta is used.

        Returns:
            out   (Tensor): Class logits   (N, num_classes).
            fused (Tensor): Fused features (N, 64).
        """
        # ResNet branch
        res_feat = self.resnet(res_b)
        res_feat = torch.flatten(res_feat, 1)
        res_feat = self.res_linear(
            res_feat, params=get_subdict(params, 'res_linear')
        )

        # ViT branch
        with torch.set_grad_enabled(
                trans_b.requires_grad or params is not None):
            vit_out = self.vit(trans_b).last_hidden_state
        vit_feat = vit_out.mean(dim=1)
        vit_feat = self.trans_linear(
            vit_feat, params=get_subdict(params, 'trans_linear')
        )

        # Multi-head self-attention fusion
        res_seq   = res_feat.unsqueeze(0)
        vit_seq   = vit_feat.unsqueeze(0)
        fused_seq, _ = self.self_attention(vit_seq, res_seq, vit_seq)
        fused     = fused_seq.squeeze(0)

        # Classification head
        out = self.fc(fused, params=get_subdict(params, 'fc'))

        return out, fused


class CoLearner(nn.Module):
    """Co-learner f_psi for gradient correction.

    Implements Eq. (co_learner_gradient):
        Delta_g_i = f_psi(phi_{i-1}', D_i^Qu)

    The network takes the query images and the flattened previous adapted
    parameters phi_{i-1}' and outputs:
        - delta_g    : gradient correction vector (grad_correction_dim,)
        - co_logits  : auxiliary classification logits for L_co

    Args:
        in_channels         (int): Input image channels (typically 3).
        num_classes         (int): Number of ways for the auxiliary head.
        hidden_size         (int): Conv feature map depth.
        grad_correction_dim (int): Dimension of the gradient correction
                                   vector = number of meta-parameters in
                                   BaseNet.
    """

    def __init__(self, in_channels: int, num_classes: int,
                 hidden_size: int, grad_correction_dim: int) -> None:
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        conv_flat_dim = hidden_size * 7 * 7

        self.cls_head = nn.Linear(conv_flat_dim, num_classes)

        self.correction_fc1 = nn.Linear(
            conv_flat_dim + grad_correction_dim, hidden_size
        )
        self.correction_fc2 = nn.Linear(hidden_size, grad_correction_dim)

    def forward(self, x: torch.Tensor,
                phi_prev_flat: torch.Tensor):
        """Compute gradient correction and auxiliary logits.

        Args:
            x             (Tensor): Query images (N, C, H, W).
            phi_prev_flat (Tensor): Flattened phi_{i-1}' (grad_correction_dim,).

        Returns:
            delta_g   (Tensor): Gradient correction (grad_correction_dim,).
            co_logits (Tensor): Auxiliary class logits (N, num_classes).
        """
        feat = self.conv1(x)
        feat = self.conv2(feat)
        feat = self.adaptive_pool(feat)
        feat = feat.view(feat.size(0), -1)

        co_logits = self.cls_head(feat)

        phi_exp = phi_prev_flat.unsqueeze(0).expand(feat.size(0), -1)
        fused   = torch.cat([feat, phi_exp], dim=1)
        delta_g = F.relu(self.correction_fc1(fused))
        delta_g = self.correction_fc2(delta_g)
        delta_g = delta_g.mean(dim=0)

        return delta_g, co_logits
