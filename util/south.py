import os, sys

import numpy as np
import SharedArray as SA
from torch.utils.data import Dataset
sys.path.append('/home/wanghe/workspace/pt-n')
from util.data_util import sa_create
from util.data_util import data_prepare


class SOUTH(Dataset):
    def __init__(self, split='train', data_root='trainval', test_area=5, voxel_size=0.04, voxel_max=None, transform=None, shuffle_index=False, loop=1):
        super().__init__()
        self.split, self.voxel_size, self.transform, self.voxel_max, self.shuffle_index, self.loop = split, voxel_size, transform, voxel_max, shuffle_index, loop

        if split == 'train':
            data_root = os.path.join(data_root, 'train_area')
        else:
            data_root = os.path.join(data_root, 'val_area')
        data_list = sorted(os.listdir(data_root))
        data_list = [item[:-4] for item in data_list]
        self.data_list = [item for item in data_list]
        for item in self.data_list:
            if not os.path.exists("/dev/shm/{}".format(item)):
                data_path = os.path.join(data_root, item + '.txt')
                data = np.loadtxt(data_path)  # xyzrgbl, N*7
                sa_create("shm://{}".format(item), data)    
        self.data_idx = np.arange(len(self.data_list))
        print("Totally {} samples in {} set.".format(len(self.data_idx), split))


    def __getitem__(self, idx):
        data_idx = self.data_idx[idx % len(self.data_idx)]
        data = SA.attach("shm://{}".format(self.data_list[data_idx])).copy()
        coord, feat, label = data[:, 0:3], data[:, 3:6], data[:, -1]-6
        coord, feat, label = data_prepare(coord, feat, label, self.split, self.voxel_size, self.voxel_max, self.transform, self.shuffle_index)
        return coord, feat, label

    def __len__(self):
        return len(self.data_idx) * self.loop

'''
if __name__ == '__main__':
    SOUTH(split='train', data_root='dataset/south/rawdata', test_area=5)
'''