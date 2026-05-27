import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from datasets import joint_transforms
from datasets.joint_transforms import JointGrayscale, JointToTensor


def get_transforms(input_dim):
    train = transforms.Compose([
        joint_transforms.JointScale(input_dim),
        joint_transforms.JointCenterCrop(input_dim, 0),
        # joint_transforms.JointRandomHorizontalFlip(),
        JointGrayscale(),
        JointToTensor()
        # joint_transforms.JointRandomRotation(90),
        # joint_transforms.JointRandomRotation(270),
        # joint_transforms.JointRandomAffine(0.2)
    ])

    val = transforms.Compose([
        joint_transforms.JointScale(input_dim),
        joint_transforms.JointCenterCrop(input_dim, 0),
        JointGrayscale(),
        JointToTensor()
    ])
    return train, val


class MyTrainDataset(Dataset):
    def __init__(self, size):
        self.size = size
        self.data = [(torch.rand(20), torch.rand(1)) for _ in range(size)]

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.data[index]
