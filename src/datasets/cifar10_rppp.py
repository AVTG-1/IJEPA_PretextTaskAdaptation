import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import random

class CIFAR10RPPP(Dataset):
    """
    Relative Patch Position Prediction on CIFAR-10.
    Splits each 32×32 image into a 4×4 grid of 8×8 patches,
    samples an anchor patch and a neighbor, and returns
    the relative-direction label (0–7).
    """

    # 8 directions: N, NE, E, SE, S, SW, W, NW
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

    def __init__(
        self,
        root,
        train=True,
        transform=None,
        grid_size=4,
        patch_size=8,
        download=True
    ):
        # Load CIFAR-10 (will download if missing)
        self.base = datasets.CIFAR10(
            root=root,
            train=train,
            transform=None,
            download=download
        )

        # Fixed grid and patch sizes for CIFAR-10
        self.grid = grid_size        # 4
        self.patch_size = patch_size  # 8

        # Always resize to 32×32 before patching
        self.transform = transform or transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
        ])

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        # Get raw image (ignore original CIFAR label)
        img, _ = self.base[idx]
        img = self.transform(img)  # [3, 32, 32]

        # Break into 4×4 grid → 16 patches of 8×8
        C, H, W = img.shape
        gs, ps = self.grid, self.patch_size
        patches = [
            img[:, i*ps:(i+1)*ps, j*ps:(j+1)*ps]
            for i in range(gs) for j in range(gs)
        ]

        # Sample anchor patch A with at least one valid neighbor
        total = gs * gs
        while True:
            a_idx = random.randrange(total)
            ai, aj = divmod(a_idx, gs)
            valid = []
            for d, (di, dj) in enumerate(self.DIRECTIONS):
                bi, bj = ai + di, aj + dj
                if 0 <= bi < gs and 0 <= bj < gs:
                    valid.append((d, bi * gs + bj))
            if valid:
                break

        dir_label, b_idx = random.choice(valid)
        patch_a = patches[a_idx]
        patch_b = patches[b_idx]

        return patch_a, patch_b, dir_label

