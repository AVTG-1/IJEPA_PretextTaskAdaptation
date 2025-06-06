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
- Pre-trained Models : checkpoints/rotation_vit_cifar10.pth, checkpoints/rppp_cifar10_vit.pth, checkpoints/multitask/multitask_epoch*.pth

## Dataset
CIFAR-10 (available in cifar-10-batches-py/)

## Multitask learning
- We have experimented with learning the respresentation of the CIFAR-10 by combining multiple pre-text tasks :- 
   1. Rotation
   2. Relative Patch Positioning
   3. Masking (Original IJEPA)
- The model has been trained with two stratergies -
   1. Direct sum of losses from the pre-text tasks
   2. Weighted sum of losses from the pre-text tasks with learned weights - $\lambda_1$, $\lambda_2$, $\lambda_3$ [Doersch et al., 2017]
- The training logs (output_MultiTask_with_WeightedSum.txt) for the weighted sum stratergy, along with the models (checkpoints/multitask/*) are stored in the repository.
