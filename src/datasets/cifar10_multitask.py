import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image

class CIFAR10MultiTask(Dataset):
    """
    For each CIFAR-10 image, produce:
      1. A random rotation (0°, 90°, 180°, 270°) and its label.
      2. One anchor + neighbor patch pair and its relative-direction label (0..7).
      3. A random 50%-mask over the 4×4 grid (16 patches) for masked-reconstruction.
    Returns a dict containing:
      - 'orig_img':   [3, 32, 32]   (normalized tensor)
      - 'rot_img':    [3, 32, 32]   (normalized tensor, rotated version)
      - 'rot_label':  int ∈ {0..3}
      - 'a_idx':      int ∈ {0..15} (anchor patch index)
      - 'b_idx':      int ∈ {0..15} (neighbor patch index)
      - 'rppp_label': int ∈ {0..7}
      - 'mask':       BoolTensor length 16 (True = masked)
    """
    # 8 possible neighbor directions: N, NE, E, SE, S, SW, W, NW
    DIRECTIONS = [
        (-1,  0),  # N
        (-1, +1),  # NE
        ( 0, +1),  # E
        (+1, +1),  # SE
        (+1,  0),  # S
        (+1, -1),  # SW
        ( 0, -1),  # W
        (-1, -1),  # NW
    ]

    def __init__(self, root, train=True, transform=None, grid_size=4, patch_size=8, mask_ratio=0.5):
        super().__init__()
        # Load CIFAR-10 (PIL images)
        self.root = root
        self.train = train
        self.cifar = datasets.CIFAR10(root=self.root, train=self.train, download=True)
        # If no transform provided, resize->tensor->normalize to [0.5,0.5,0.5]
        self.transform = transform or transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5)),
        ])

        self.grid = grid_size      # 4
        self.patch_size = patch_size  # 8 (since 32/4 = 8)
        self.mask_ratio = mask_ratio  # fraction of patches to mask

    def __len__(self):
        return len(self.cifar)

    def __getitem__(self, idx):
        # 1) Load PIL image (ignore CIFAR label)
        pil_img, _ = self.cifar[idx]

        # 2) Create original (“orig_img”) and rotated version (“rot_img”)
        orig_tensor = self.transform(pil_img)  # [3,32,32]
        # Pick random rotation label ∈ {0,1,2,3} → 0°,90°,180°,270°
        rot_label = random.choice([0, 1, 2, 3])
        rotated_pil = pil_img.rotate(90 * rot_label)
        rot_tensor = self.transform(rotated_pil)  # [3,32,32]

        # 3) Prepare RPPP: split orig_tensor into 4×4 patches of size 8×8
        C, H, W = orig_tensor.shape  # (3,32,32)
        gs, ps = self.grid, self.patch_size
        patches = [
            orig_tensor[:, i*ps:(i+1)*ps, j*ps:(j+1)*ps]
            for i in range(gs) for j in range(gs)
        ]
        total = gs * gs  # 16

        # Sample an anchor patch index that has at least one valid neighbor
        while True:
            a_idx = random.randrange(total)
            ai, aj = divmod(a_idx, gs)
            valid_neighbors = []
            for d, (di, dj) in enumerate(self.DIRECTIONS):
                bi, bj = ai + di, aj + dj
                if 0 <= bi < gs and 0 <= bj < gs:
                    valid_neighbors.append((d, bi * gs + bj))
            if valid_neighbors:
                break

        # Randomly pick one valid neighbor
        rppp_label, b_idx = random.choice(valid_neighbors)
        # patch_a = patches[a_idx]   # [3,8,8]  (not returned directly)
        # patch_b = patches[b_idx]   # [3,8,8]

        # 4) Prepare masked-reconstruction mask over 16 patches
        num_patches = total
        num_mask = int(self.mask_ratio * num_patches)
        all_indices = list(range(num_patches))
        mask_indices = random.sample(all_indices, num_mask)
        mask = torch.zeros(num_patches, dtype=torch.bool)
        mask[mask_indices] = True
        # (True means “this patch is masked”)

        return {
            'orig_img':   orig_tensor,   # [3,32,32]
            'rot_img':    rot_tensor,    # [3,32,32]
            'rot_label':  torch.tensor(rot_label, dtype=torch.long),
            'a_idx':      torch.tensor(a_idx, dtype=torch.long),
            'b_idx':      torch.tensor(b_idx, dtype=torch.long),
            'rppp_label': torch.tensor(rppp_label, dtype=torch.long),
            'mask':       mask           # BoolTensor length=16
        }

def get_cifar10_dataloaders(batch_size=128, root='./cifar-10-batches-py/..'):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5)),
    ])
    train_dataset = CIFAR10MultiTask(root=root, train=True, transform=transform)
    val_dataset = CIFAR10MultiTask(root=root, train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader

