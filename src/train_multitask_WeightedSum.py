import os
import torch
import torch.nn as nn
import torch.optim as optim

from datasets.cifar10_multitask import CIFAR10MultiTask, get_cifar10_dataloaders
from models.multi_task_vit import MultiTaskViT
from models.patch_decoder import PatchDecoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
BATCH_SIZE = 128
NUM_EPOCHS = 50
LR = 1e-4

# DataLoaders
train_loader, val_loader = get_cifar10_dataloaders(BATCH_SIZE)

# Models
model = MultiTaskViT(img_size=32, patch_size=8, in_chans=3).to(device)
decoder = PatchDecoder(embed_dim=192, patch_size=8, out_chans=3).to(device)

# Checkpoint directory
checkpoint_dir = "checkpoints/multitask"
os.makedirs(checkpoint_dir, exist_ok=True)

# Learnable log variances (uncertainty-based weighting)
log_var_rot   = torch.nn.Parameter(torch.zeros(1, device=device))
log_var_rppp  = torch.nn.Parameter(torch.zeros(1, device=device))
log_var_recon = torch.nn.Parameter(torch.zeros(1, device=device))

# Optimizer (include model + decoder + log_vars)
optimizer = optim.Adam(
    list(model.parameters()) +
    list(decoder.parameters()) +
    [log_var_rot, log_var_rppp, log_var_recon],
    lr=LR
)

# Loss functions
criterion_rot   = nn.CrossEntropyLoss()
criterion_rppp  = nn.CrossEntropyLoss()
criterion_recon = nn.MSELoss()

# Training loop
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0.0
    correct_rot = 0
    correct_rppp = 0
    total_samples = 0

    for batch in train_loader:
        # batch is a dict with keys: 
        # 'orig_img', 'rot_img', 'rot_label', 'a_idx', 'b_idx', 'rppp_label', 'mask'
        # Move all tensors to device:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        optimizer.zero_grad()

        # Forward pass through MultiTaskViT
        # The model's forward expects exactly that dict:
        outputs = model(batch)
        # outputs is another dict containing:
        # 'rot_logits', 'rppp_logits', 'pred_masked' [B,N_mask,D], 'tgt_masked' [B,N_mask,D]

        # --- Rotation Loss & Acc ---
        rot_logits = outputs['rot_logits']             # [B,4]
        rot_labels = batch['rot_label']                # [B]
        loss_rot = criterion_rot(rot_logits, rot_labels)
        preds_rot = rot_logits.argmax(dim=1)
        correct_rot += (preds_rot == rot_labels).sum().item()

        # --- RPPP Loss & Acc ---
        rppp_logits = outputs['rppp_logits']           # [B,8]
        rppp_labels = batch['rppp_label']              # [B]
        loss_rppp = criterion_rppp(rppp_logits, rppp_labels)
        preds_rppp = rppp_logits.argmax(dim=1)
        correct_rppp += (preds_rppp == rppp_labels).sum().item()

        # --- Reconstruction Loss ---
        # predicted masked embeddings vs target masked embeddings
        pred_embeds = outputs['pred_masked']           # [B, N_mask, D]
        tgt_embeds  = outputs['tgt_masked']            # [B, N_mask, D]
        # We need to decode pred_embeds → pixel patches, then compute MSE vs actual pixel patches.
        # But CIFAR10MultiTask does NOT give us the “ground-truth pixel patches” directly.
        # Instead, we can re-extract those patches from batch['orig_img']:
        B = batch['orig_img'].size(0)
        mask = batch['mask']                            # [B,16] BoolTensor
        patch_size = decoder.patch_size                  # 8
        # 1) Decode predicted embeddings into pixel patches:
        #    Flatten [B, N_mask, D] to [B * N_mask, D] first:
        B_effective, N_mask, D = pred_embeds.shape
        pred_flat = pred_embeds.view(B_effective * N_mask, D)  # [B*N_mask, D]
        decoded_patches = decoder(pred_flat)                   # [B*N_mask, 3, 8, 8]

        # 2) Gather the ground-truth pixel patches from orig_img:
        #    a) Unfold orig_img to get all 16 patches per image:
        #    orig_img: [B, 3, 32, 32]
        orig_imgs = batch['orig_img']  # [B, 3, 32, 32]
        #    Unfold:  (kernel_size=8, stride=8) → shape [B, 3, 4, 4, 8, 8]
        patches = orig_imgs.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        #    Now permutation and reshape to get [B, 16, 3, 8, 8]:
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.view(B, -1, 3, patch_size, patch_size)  # [B,16,3,8,8]

        #    b) For each sample i in [0..B-1], select the masked indices:
        gt_list = []
        for i in range(B):
            masked_idxs = torch.where(mask[i])[0]           # e.g. [i1, i2, ...], length = N_mask
            gt_list.append(patches[i, masked_idxs, :, :, :])  # [N_mask, 3, 8, 8]
        # Stack to [B, N_mask, 3, 8, 8]
        gt_masked = torch.stack(gt_list, dim=0)             # [B, N_mask, 3, 8, 8]
        # Flatten to [B*N_mask, 3, 8, 8]
        gt_flat = gt_masked.view(B * N_mask, 3, patch_size, patch_size)

        # 3) Compute MSE between decoded_patches ([B*N_mask, 3, 8, 8]) vs gt_flat:
        loss_recon = criterion_recon(decoded_patches, gt_flat)

        # --- Combine Losses with Uncertainty-based Weights ---
        precision_rot   = torch.exp(-log_var_rot)
        precision_rppp  = torch.exp(-log_var_rppp)
        precision_recon = torch.exp(-log_var_recon)

        loss = (
            precision_rot   * loss_rot   + log_var_rot +
            precision_rppp  * loss_rppp  + log_var_rppp +
            precision_recon * loss_recon + log_var_recon
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_samples += B

    # Epoch‐level metrics
    avg_loss = total_loss / total_samples
    acc_rot = 100.0 * correct_rot / total_samples
    acc_rppp = 100.0 * correct_rppp / total_samples

    # Print epoch stats
    print(
        f"Epoch {epoch+1:02d} | Avg Loss: {avg_loss:.4f} | "
        f"Rot Acc: {acc_rot:.2f}% | RPPP Acc: {acc_rppp:.2f}%"
    )
    print(
        f"LogVars: rot={log_var_rot.item():.4f}, "
        f"rppp={log_var_rppp.item():.4f}, "
        f"recon={log_var_recon.item():.4f}"
    )

    # Save checkpoint every 10 epochs
    if (epoch + 1) % 10 == 0:
        ckpt = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'log_var_rot':   log_var_rot.detach().cpu().item(),
            'log_var_rppp':  log_var_rppp.detach().cpu().item(),
            'log_var_recon': log_var_recon.detach().cpu().item(),
        }
        torch.save(ckpt, os.path.join(checkpoint_dir, f"multitask_epoch{epoch+1}.pth"))
        print(f"Saved checkpoint: {checkpoint_dir}/multitask_epoch{epoch+1}.pth")

print("Multitask training complete.")
