import numpy as np
import os
import shutil

# train_directory = 'data/dad/vgg16_features/training/'
# train_directory = 'data/dad/correct/'
train_directory = 'data/sample/dad/vgg16_features/training/'
trainfiles_root= os.listdir(train_directory)

positive=[]
negative=[]
for i in trainfiles_root:
    file_name = train_directory +i
    b = np.load(file_name, allow_pickle=True)
    label = b['labels']
    if label[1]> 0:
        positive.append(i)
    else:
        negative.append(i)

print(len(positive))

source_dir = train_directory
# destination_positive = 'copy_destination/positive/'
destination_positive= 'data/sample/dad/vgg16_features/positive/'
count = 0
for i in positive:
    source_file_name = source_dir+ i
    shutil.move(source_file_name, destination_positive)
    count+=1
print('total file moved =', count )
