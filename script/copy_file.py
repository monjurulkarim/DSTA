import numpy as np
import shutil
import glob

source_dir = 'data/dad/vgg16_features/testing/*'
destination = 'data/dad/copied_file_iteration_1/'


count = 0
for file in glob.glob(source_dir):
    print(file)
#     print(str(file[16:22]))
    # d = str(file[32:44])
    d_1 = str(file[32:37])
    d_2 = str(file[38:44])
    d = d_1 + d_2
    print(d)
    # print(d)
    # print(destination + str(file[32:36]) + '_1' + str(file[37:43])+ '.npz')
    # copied_file = destination + str(file[32:36]) + '_1' + str(file[37:43])+ '.npz'
    copied_file = destination + d + '.npz'
    print(str(file[32:36]) + '_0' + str(file[37:43])+ '.npz')
    shutil.copyfile(file, copied_file)
    # print('file copied : ', file)
    count+=1
