import h5py
import numpy as np
import os

# data_root = '/home/wanghe/workspace/pt-n/dataset/south/rawdata'
# data_list = sorted(os.listdir(data_root))

start_id = 0
num = np.random.randint(200000, 600000, 10)
for i in range(len(num)):
    coord = np.random.rand(num[i], 3)
    coord = coord*10000 + 440000
    feat = np.random.randint(0, 255, (num[i], 3))
    label = np.random.randint(6, 9, (num[i], 1))
    print(coord.shape, feat.shape, label.shape)
    data = np.concatenate((coord, feat, label), axis = 1)
    print(data.shape)

    s = "%02d" % start_id
    np.savetxt('/home/wanghe/workspace/pt-n/dataset/south/trainval_fullarea/data{}.txt'.format(s), data)
    print('saved data{}.txt'.format(s))
    start_id += 1
print('Done')





'''start_id = 0
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
print('Done!')'''