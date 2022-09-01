import h5py
import numpy as np
import os

data_root = '/home/wanghe/workspace/pt-n/dataset/south/rawdata'
data_list = sorted(os.listdir(data_root))
start_id = 0
for item in data_list:
    filename = os.path.join(data_root, item)
    read_data = np.loadtxt(filename)   # raw data path
    data = read_data[:, 0:6]
    label = read_data[:, -1]       # -1/-2
    length = len(label)
    index = 0
    print('start prepare data file -- {}'.format(item))
    while index+4096 <= length:
        s = "%06d" % start_id
        f = h5py.File('/home/wanghe/workspace/pt-n/dataset/south/data/{}.h5'.format(s),'w')    # save path
        f['data'] = data[index:index+4096, :]
        f['label'] = label[index:index+4096]
        f.close()
        print('save prepared data file {}.h5'.format(s))
        with open('/home/wanghe/workspace/pt-n/dataset/south/list/scene.txt', encoding="utf-8",mode="a") as file:   #save list.txt path
            file.write('{}.h5\n'.format(s))
        start_id += 1
        index += 4096
    print('Raw data file {} finished, the next scene_id should be {}'.format(item, start_id))
print('Done!')