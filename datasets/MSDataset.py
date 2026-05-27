import os
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
from torchvision.datasets.folder import default_loader
from torchvision.transforms import ToTensor


def _make_dataset(dir, patient_name=None):
    images = []
    annot_dir = str(dir) + 'annot'
    for root, _, fnames in sorted(os.walk(annot_dir)):
        if patient_name is None or root == os.path.join(annot_dir, patient_name):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                im = np.asanyarray(Image.open(path))
                if im.sum() > 0:
                    images.append(path)
    return images


class MSDataset(data.Dataset):
    def __init__(self, root, split='train', input_dim=128, mean=0.1026, std=0.0971,
                 joint_transform=None, target_transform=None,
                 loader=default_loader, seq_size=1, size=None, patient_name=None):
        self.root            = str(root)
        self.input_dim       = input_dim
        self.split           = split
        self.joint_transform = joint_transform
        self.loader          = loader
        self.mean            = mean
        self.std             = std
        self.seq_size        = seq_size
        self.to_tensor       = ToTensor()
        self.imgs            = _make_dataset(
            os.path.join(self.root, self.split), patient_name)

    def __len__(self):
        return len(self.imgs) - self.seq_size + 1

    def __getitem__(self, index):
        pil_imgs    = []
        pil_targets = []

        for i in range(self.seq_size):
            path = self.imgs[index + i]

            # Linux-compatible path manipulation
            fname    = os.path.basename(path)
            dir_path = os.path.dirname(path)
            name_img = fname.replace('mask1', 'flair_pp')
            img_dir  = dir_path.replace(self.split + 'annot', self.split)
            path_img = os.path.join(img_dir, name_img)

            pil_imgs.append(self.loader(path_img))
            pil_targets.append(self.loader(path))

        if self.joint_transform is not None:
            transformed = self.joint_transform(pil_imgs + pil_targets)
        else:
            transformed = pil_imgs + pil_targets

        imgs    = []
        targets = []

        for i in range(self.seq_size):
            img = transformed[i]
            if not isinstance(img, torch.Tensor):
                img = self.to_tensor(img)
            m1 = img.min().item()
            m2 = img.max().item()
            if m1 != m2:
                img = (img - m1) / (m2 - m1)
                img = (img - self.mean) / self.std
            imgs.append(img)  # [1, H, W]

            tgt = transformed[self.seq_size + i]
            if not isinstance(tgt, torch.Tensor):
                tgt = self.to_tensor(tgt)
            tgt = (tgt > 0.2).float()  # [1, H, W]
            targets.append(tgt)

        return torch.cat(imgs, dim=0), torch.cat(targets, dim=0)
