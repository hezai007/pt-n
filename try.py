from cProfile import label
import os
import h5py  #导入工具包
import numpy as np
import SharedArray as SA
from torch.utils.data import Dataset
from util.data_util import sa_create

split = 'train'
data_root='/home/wanghe/workspace/pt-n/dataset/south/trainval/train'
data_list = sorted(os.listdir(data_root))
# data_list = [item[:-4] for item in data_list if not 'Area_' in item]
print(data_list)
data_list = [item for item in data_list]
print('end')
for item in data_list:
    if not os.path.exists("/dev/shm/{}".format(item)):
        data_path = os.path.join(data_root, item)
        data = np.load(data_path)  # xyzrgbl, N*7
        sa_create("shm://{}".format(item), data)    
data_idx = np.arange(len(data_list))
print("Totally {} samples in {} set.".format(len(data_idx), split))


trya = np.arange(15).reshape(5,3)
print(trya[0:3, :])
#HDF5的读取：
f = h5py.File('/home/wanghe/workspace/pt-n/dataset/s3dis/trainval/00000000.h5','r')   #打开h5文件
f.keys()                            #可以查看所有的主键
print(f.keys())
a = f['data'][:]                    #取出主键为data的所有的键值
print(a)
print(a.size)
f.close()


filename = 'data'
read_data = np.loadtxt('/home/wanghe/workspace/pt-n/dataset/try/{}.txt'.format(filename))   # raw data path
data = read_data[:, 0:6]
label = read_data[:, -1]       # -1/-2
length = len(label)
index = 0
start_id = 0
print('start prepare data file -- {}'.format(filename))
while index+4096 <= length:
    s = "%06d" % start_id
    f = h5py.File('/home/wanghe/workspace/pt-n/dataset/try/{}.h5'.format(s),'w')    # save path
    f['data'] = data[index:index+4096, :]
    f['label'] = label[index:index+4096]
    f.close()
    print('save prepared data file {}.h5'.format(s))
    with open('/home/wanghe/workspace/pt-n/dataset/try/list.txt', encoding="utf-8",mode="a") as file:   #save list.txt path
        file.write('{}.h5\n'.format(s))
    start_id += 1
    index += 4096
print('Done!')


        



