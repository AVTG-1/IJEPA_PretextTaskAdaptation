import torch
import torch.nn as nn
import torch.nn.functional as F
from models.vision_transformer import VisionTransformer

class MultiTaskViT(nn.Module):
    """
    A shared-ViT encoder with three “heads”:
     1. Rotation prediction (4-way).
     2. RPPP (8-way) on two patch embeddings.
     3. Masked-reconstruction: predictor network vs. EMA “target” encoder.
    """
    def __init__(self,
                 img_size=32,
                 patch_size=8,
                 in_chans=3,
                 embed_dim=192,
                 depth=6,
                 num_heads=3,
                 mlp_ratio=4.0,
                 ema_momentum=0.999):
        super().__init__()

        # --- Shared ViT Encoder ---
        self.encoder = VisionTransformer(
            img_size=[img_size],
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=True
        )
        num_patches = self.encoder.patch_embed.num_patches  # should be 16 for 32×32 with patch_size=8

        # --- Rotation Head ---
        # Global average over patch embeddings → 4 logits
        self.rot_head = nn.Linear(embed_dim, 4)

        # --- RPPP Head ---
        # Concatenate two patch embeddings (2×D) → MLP → 8 logits
        self.rppp_mlp = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 8)
        )

        # --- Masked-Reconstruction Predictor ---
        # Simplest: Given an average of visible patch embeddings, predict each masked embedding via a linear layer.
        self.recon_pred = nn.Linear(embed_dim, embed_dim)

        # --- Target Encoder (EMA copy, frozen) ---
        self.target_encoder = VisionTransformer(
            img_size=[img_size],
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=True
        )
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # Initialize target_encoder weights ← encoder weights
        for tgt_p, src_p in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            tgt_p.data.copy_(src_p.data)

        self.ema_momentum = ema_momentum

    @torch.no_grad()
    def _update_target_encoder(self):
        # EMA update: θ_target = m·θ_target + (1−m)·θ
        for tgt_p, src_p in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            tgt_p.data.mul_(self.ema_momentum).add_((1.0 - self.ema_momentum) * src_p.data)

    def update_ema(self):
        """Call this right after each optimizer.step()"""
        self._update_target_encoder()

    def forward(self, batch):
        """
        Expects batch to contain:
          - orig_img:   [B,3,32,32]
          - rot_img:    [B,3,32,32]
          - rot_label:  [B] (long)
          - a_idx:      [B] (long, 0..15)
          - b_idx:      [B] (long, 0..15)
          - rppp_label: [B] (long, 0..7)
          - mask:       [B,16] (BoolTensor)
        Returns a dict with:
          - rot_logits:  [B,4]
          - rppp_logits: [B,8]
          - pred_masked: [B, N_mask, D]
          - tgt_masked:  [B, N_mask, D]
        """
        B = batch['orig_img'].size(0)
        device = batch['orig_img'].device

        # ------ Rotation branch ------
        # Encode rot_img → [B, N=16, D]
        rot_feats = self.encoder(batch['rot_img'])       # [B,16,D]
        rot_pooled = rot_feats.mean(dim=1)                # [B,D]
        rot_logits = self.rot_head(rot_pooled)            # [B,4]

        # ------ Shared Encode orig_img for RPPP & Reconstruction ------
        orig_feats = self.encoder(batch['orig_img'])      # [B,16,D]

        # --- RPPP branch ---
        # a_idx, b_idx: indices into the 16 patch embeddings
        a_idx = batch['a_idx']  # [B]
        b_idx = batch['b_idx']  # [B]
        # Gather patch embeddings
        emb_a = orig_feats[torch.arange(B, device=device), a_idx, :]  # [B,D]
        emb_b = orig_feats[torch.arange(B, device=device), b_idx, :]  # [B,D]
        rppp_input = torch.cat([emb_a, emb_b], dim=1)                 # [B, 2D]
        rppp_logits = self.rppp_mlp(rppp_input)                        # [B,8]

        # --- Masked-Reconstruction branch ---
        mask = batch['mask']  # [B,16], BoolTensor
        # 1) Target embeddings (no grad)
        with torch.no_grad():
            tgt_feats = self.target_encoder(batch['orig_img'])  # [B,16,D]
            # Extract masked embeddings from tgt_feats
            all_tgt = []
            for i in range(B):
                masked_idxs = torch.where(mask[i])[0]        # e.g. [i1, i2, ...]
                all_tgt.append(tgt_feats[i, masked_idxs, :]) # [N_mask, D]
            # Assume N_mask is constant (since mask_ratio fixed), so stack:
            tgt_masked = torch.stack(all_tgt, dim=0)         # [B, N_mask, D]

        # 2) Prediction: use only the visible embeddings from orig_feats
        all_preds = []
        for i in range(B):
            unmasked_idxs = torch.where(~mask[i])[0]            # indices where mask=False
            emb_vis = orig_feats[i, unmasked_idxs, :]            # [N_vis, D]
            avg_vis = emb_vis.mean(dim=0, keepdim=True)          # [1, D]
            # Predict same embedding for each masked slot:
            #   pred = recon_pred(avg_vis) → [1, D], repeat to [N_mask, D]
            num_mask = tgt_masked.size(1)                        # N_mask
            pred = self.recon_pred(avg_vis)                       # [1,D]
            pred = pred.repeat(num_mask, 1)                       # [N_mask,D]
            all_preds.append(pred)
        pred_masked = torch.stack(all_preds, dim=0)               # [B, N_mask, D]

        return {
            'rot_logits':  rot_logits,     # [B,4]
            'rppp_logits': rppp_logits,    # [B,8]
            'pred_masked': pred_masked,    # [B, N_mask, D]
            'tgt_masked':  tgt_masked      # [B, N_mask, D]
        }
