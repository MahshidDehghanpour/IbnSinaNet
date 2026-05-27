import os
from os.path import isdir, join

import numpy as np
from PIL import Image
from numpy import ndarray


def list_ave(lst):
    if len(lst) == 0:
        return 0
    return sum(lst) / len(lst)

def get_unique_file(directory, num=1):
    files = sorted([f for f in os.listdir(directory) if isdir(join(directory, f))])
    for f in files:
        if f.startswith(f'{num:03d}'):
            num += 1
    return f'{num:03d}'


def numpy_0_1_to_pil(array: ndarray):
    if len(array.shape) == 3 and array.shape[2] == 1:
        array = np.squeeze(array, axis=2)
    return Image.fromarray((array * 255).astype(np.uint8))


def numpy_0_255_to_pil(array: ndarray):
    if len(array.shape) == 3 and array.shape[2] == 1:
        array = np.squeeze(array, axis=2)
    return Image.fromarray(array.astype(np.uint8))

def torch_to_numpy(tensor):
    return tensor.permute(1, 2, 0).cpu().detach().numpy()

def save_tensor_batch(tensor, directory, file_name, is_0_255=False, normalize=(0,1), max_images=-1):
    batch = tensor.size()[0]
    for i in range(batch):
        if i == max_images:
            break
        t = tensor[i, :, :, :]
        arr = torch_to_numpy(t)
        arr = (arr * normalize[1] + normalize[0])
        if is_0_255:
            img = numpy_0_255_to_pil(arr)
        else:
            img = numpy_0_1_to_pil(arr)
        if isinstance(file_name, list):
            name = f'{file_name[i]}.png'
        else:
            name = f'{file_name}-{i}.png'
        img.save(os.path.join(directory, name))