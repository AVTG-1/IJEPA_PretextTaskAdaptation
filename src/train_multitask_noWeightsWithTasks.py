"""

This code doesn't consider a weighted sum of losses for multi-task training.
The final loss is a simple sum of the individual task losses.

"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets.cifar10_multitask import CIFAR10MultiTask
from models.multi_task_vit import MultiTaskViT
from models.patch_decoder import PatchDecoder

# Set CUDA_VISIBLE_DEVICES if needed, or let user specify
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_multitask(
    root="./",
    batch_size=128,
    	# aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
    epochs=50,
    	# Number of epochs have been altered for testing
    lr=1e-3,
    mask_ratio=0.5,
    ema_momentum=0.999
):
    # 1) Dataset & DataLoader
    train_set = CIFAR10MultiTask(
        root=root,
        train=True,
        transform=None,      # default transform handles resize+normalize
        grid_size=4,
        patch_size=8,
        mask_ratio=mask_ratio
    )
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # 2) Instantiate Model
    model = MultiTaskViT(
        img_size=32,
        patch_size=8,
        in_chans=3,
        embed_dim=192,
        depth=6,
        num_heads=3,
        mlp_ratio=4.0,
        ema_momentum=ema_momentum
    ).to(device)

    # 3) Loss functions
    rot_criterion   = nn.CrossEntropyLoss()
    rppp_criterion  = nn.CrossEntropyLoss()
    recon_criterion = nn.MSELoss()

    # 4) Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)

    # 5) (Optional) Decoder for saving reconstructed patches (not used in training)
    #    You can train this separately after multi-task training
    decoder = PatchDecoder(embed_dim=192, patch_size=8, out_chans=3).to(device)

    # Create a folder for checkpoints if not exist
    os.makedirs("checkpoints/multitask", exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_rot_loss   = 0.0
        total_rppp_loss  = 0.0
        total_recon_loss = 0.0
        correct_rot = 0
        correct_rppp = 0
        total_samples = 0

        for batch in train_loader:
            # Move all tensors in batch to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            # Forward pass
            outputs = model(batch)

            # 1) Rotation loss + accuracy
            rot_logits = outputs['rot_logits']            # [B,4]
            rot_loss   = rot_criterion(rot_logits, batch['rot_label'])
            rot_preds  = rot_logits.argmax(dim=1)
            rot_acc    = (rot_preds == batch['rot_label']).sum().item()

            # 2) RPPP loss + accuracy
            rppp_logits = outputs['rppp_logits']          # [B,8]
            rppp_loss   = rppp_criterion(rppp_logits, batch['rppp_label'])
            rppp_preds  = rppp_logits.argmax(dim=1)
            rppp_acc    = (rppp_preds == batch['rppp_label']).sum().item()

            # 3) Reconstruction loss (MSE between pred_masked vs tgt_masked)
            pred = outputs['pred_masked']                 # [B, N_mask, D]
            tgt  = outputs['tgt_masked']                  # [B, N_mask, D]
            recon_loss = recon_criterion(pred, tgt)

            # Combine (equal weights for now)
            total_loss = rot_loss + rppp_loss + recon_loss

            # Backprop
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            model.update_ema()

            # Statistics
            B_eff = batch['orig_img'].size(0)
            total_rot_loss   += rot_loss.item() * B_eff
            total_rppp_loss  += rppp_loss.item() * B_eff
            total_recon_loss += recon_loss.item() * B_eff
            correct_rot   += rot_acc
            correct_rppp  += rppp_acc
            total_samples += B_eff

        # Compute epoch averages
        avg_rot_loss   = total_rot_loss / total_samples
        avg_rppp_loss  = total_rppp_loss / total_samples
        avg_recon_loss = total_recon_loss / total_samples
        rot_acc_epoch  = 100.0 * correct_rot / total_samples
        rppp_acc_epoch = 100.0 * correct_rppp / total_samples

        print(
            f"Epoch {epoch:02d} | "
            f"Rot Loss: {avg_rot_loss:.4f}, Rot Acc: {rot_acc_epoch:.2f}% | "
            f"RPPP Loss: {avg_rppp_loss:.4f}, RPPP Acc: {rppp_acc_epoch:.2f}% | "
            f"Recon Loss: {avg_recon_loss:.4f}"
        )

        # Save a checkpoint for this epoch
        ckpt = {
            'epoch': epoch,
            'encoder_state': model.encoder.state_dict(),
            'rot_head':       model.rot_head.state_dict(),
            'rppp_mlp':       model.rppp_mlp.state_dict(),
            'recon_pred':     model.recon_pred.state_dict(),
            'optimizer':      optimizer.state_dict()
        }

        torch.save(ckpt, f"checkpoints/multitask/multitask_epoch{epoch}.pth")

    print("Multitask training complete.")
    # (Optional) After training, you can train `decoder` on (tgt_masked_embeddings vs. original pixel patches)
    return model, decoder

if __name__ == "__main__":
    # You can adjust root, batch_size, epochs, lr, etc., via command-line args if desired
    train_multitask(
        root="./",
        batch_size=128,
        epochs=10,
        lr=1e-3,
        mask_ratio=0.5,
        ema_momentum=0.999
    )
