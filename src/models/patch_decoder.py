import torch
import torch.nn as nn

class PatchDecoder(nn.Module):
    """
    Given a “masked-patch embedding” of dimension D, decode it back to a 3×8×8 pixel patch.
    (In CIFAR-10 with 4×4 grid, patch_size=8.)
    """
    def __init__(self, embed_dim=192, patch_size=8, out_chans=3):
        super().__init__()
        self.patch_size = patch_size
        self.out_chans = out_chans
        self.output_dim = out_chans * patch_size * patch_size  # e.g. 3*8*8 = 192

        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, self.output_dim)
        )

    def forward(self, patch_embs):
        """
        patch_embs: [B * N_mask, D]
        Returns:   pixel patches of shape [B * N_mask, 3, 8, 8]
        """
        x = self.fc(patch_embs)                            # [B*N_mask, 3*8*8]
        x = x.view(-1, self.out_chans, self.patch_size, self.patch_size)
        return x
