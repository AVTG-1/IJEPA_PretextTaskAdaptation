import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets.cifar10_rotation import CIFAR10Rotation
from models.rotation_model import ViTRotationModel
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparams
batch_size = 128
epochs = 10
lr = 1e-3

# Data
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_set = CIFAR10Rotation(root="./", train=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

# Model
model = ViTRotationModel(
    img_size=[224],
    patch_size=16,
    in_chans=3,
    embed_dim=384,
    depth=6,
    num_heads=6,
    mlp_ratio=4.0,
    qkv_bias=True,
    num_classes=4
).to(device)

# Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Train
for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()

    acc = 100 * correct / len(train_loader.dataset)
    print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}, Acc = {acc:.2f}%")

# Save the trained model
save_path = "checkpoints/rotation_vit_cifar10.pth"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
torch.save(model.state_dict(), save_path)
print(f"Model saved to {save_path}")

