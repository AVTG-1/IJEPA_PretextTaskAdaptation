# I-JEPA with Pretext Tasks

Adaptation of Facebook's I-JEPA for self-supervised learning with pretext tasks on CIFAR-10.

## Key Features
- **Pretext Tasks**:
  - Rotation Prediction (0°, 90°, 180°, 270°)
  - Relative Patch Position Prediction (RPPP)
- Lightweight ViT encoder
- CIFAR-10 focused implementation

## Usage
1. **Rotation Task**:
   ```bash
   python src/train_rotation.py
2. **RPP Task**:
   ```bash
   python src/train_rppp.py

## Outputs
- Training Logs : output_Rotation.txt, output_RelativePatch.txt
- Pre-trained Models : checkpoints/rotation_vit_cifar10.pth, checkpoints/rppp_cifar10_vit.pth

## Dataset
CIFAR-10 (available in cifar-10-batches-py/)
