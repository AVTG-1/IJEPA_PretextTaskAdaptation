import torch
import torch.nn as nn
from models.vision_transformer import VisionTransformer

class ViTRotationModel(nn.Module):
    def __init__(
        self,
        img_size=[224],
        patch_size=16,
        in_chans=3,
        embed_dim=384,
        depth=6,
        num_heads=6,
        mlp_ratio=4.0,
        qkv_bias=True,
        num_classes=4
    ):
        super().__init__()
        # The base ViT encoder (returns patch embeddings of shape [B, N, D])
        self.encoder = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias
        )
        # Classification head: project the pooled embedding to 4 classes
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # x: [B, 3, H, W] → encoder → [B, N, D]
        x = self.encoder(x)           
        # Global average pool over patches:
        x = x.mean(dim=1)            
        # Classification logits:
        x = self.head(x)             
        return x

