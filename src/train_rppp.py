import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from datasets.cifar10_rppp import CIFAR10RPPP
from models.vision_transformer import VisionTransformer

# ---------------------------------------------------
# 1) Monkey-patch VisionTransformer to skip interpolation
# ---------------------------------------------------
def _no_interp(self, x, pos_embed):
    # Simply return the stored pos_embed (shape [1, N, D])
    return pos_embed

VisionTransformer.interpolate_pos_encoding = _no_interp
# ---------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 128
epochs     = 1000
lr         = 1e-3
grid_size  = 4
patch_size = 8   # 32 / 4

# DataLoader
train_set = CIFAR10RPPP(
    root="./",
    train=True,
    transform=None,
    grid_size=grid_size,
    patch_size=patch_size,
    download=True
)
train_loader = DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

# Model: concatenates two patches → projects 6→3 channels → ViT → 8-way head
class RPPPModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 6-channel input (two RGB patches) → back to 3 channels
        self.proj = nn.Conv2d(6, 3, kernel_size=1)
        # Small ViT for 32×32 input, 8×8 patches → 16 tokens
        self.vit = VisionTransformer(
            img_size=[32],
            patch_size=8,
            in_chans=3,
            embed_dim=192,
            depth=6,
            num_heads=3,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.1,
            attn_drop_rate=0.1,
            drop_path_rate=0.1
        )
        self.head = nn.Linear(192, 8)

    def forward(self, a, b):
        # a, b each [B,3,32,32] → concat → [B,6,32,32]
        x = torch.cat([a, b], dim=1)
        x = self.proj(x)           # [B,3,32,32]
        x = self.vit(x)            # [B, num_patches=16, 192]
        x = x.mean(dim=1)          # Global average → [B,192]
        return self.head(x)        # [B,8]

model = RPPPModel().to(device)

# Loss & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

# Training loop
for epoch in range(1, epochs+1):
    model.train()
    total_loss = 0.0
    correct    = 0

    for a, b, label in train_loader:
        a, b, label = a.to(device), b.to(device), label.to(device)
        optimizer.zero_grad()
        logits = model(a, b)
        loss   = criterion(logits, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * a.size(0)
        correct    += (logits.argmax(dim=1) == label).sum().item()

    avg_loss = total_loss / len(train_loader.dataset)
    acc      = 100.0 * correct / len(train_loader.dataset)
    print(f"[Epoch {epoch}/{epochs}] Loss: {avg_loss:.4f}  Acc: {acc:.2f}%")

save_path = "checkpoints/rppp_cifar10_vit.pth"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
torch.save(model.state_dict(), save_path)
print(f"Model weights saved to {save_path}")

