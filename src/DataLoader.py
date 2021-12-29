from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset
#import networkx
#import itertools


class DADDataset(Dataset):
    def __init__(self, data_path, feature, phase='training', toTensor=False, device=torch.device('cuda'), vis=False):
        self.data_path = os.path.join(data_path, feature + '_features')
        self.feature = feature
        self.phase = phase
        self.toTensor = toTensor
        self.device = device
        self.vis = vis
        self.n_frames = 100
        self.n_obj = 19
        self.fps = 20.0
        self.dim_feature = self.get_feature_dim(feature)

        filepath = os.path.join(self.data_path, phase)
        self.files_list = self.get_filelist(filepath)

    def __len__(self):
        data_len = len(self.files_list)
        return data_len

    def get_feature_dim(self, feature_name):
        if feature_name == 'vgg16':
            return 4096
        elif feature_name == 'res101':
            return 2048
        else:
            raise ValueError

    def get_filelist(self, filepath):
        assert os.path.exists(filepath), "Directory does not exist: %s"%(filepath)
        file_list = []
        for filename in sorted(os.listdir(filepath)):
            file_list.append(filename)
        return file_list

    def __getitem__(self, index):
        data_file = os.path.join(self.data_path, self.phase, self.files_list[index])
        assert os.path.exists(data_file)
        try:
            data = np.load(data_file)
            features = data['data']  # 100 x 20 x 4096
            labels = data['labels']  # 2
            detections = data['det']  # 100 x 19 x 6
        except:
            raise IOError('Load data error! File: %s'%(data_file))
        if labels[1] > 0:
            toa = [90.0]
        else:
            toa = [self.n_frames + 1]

        if self.toTensor:
            features = torch.Tensor(features).to(self.device)         #  100 x 20 x 4096
            labels = torch.Tensor(labels).to(self.device)
            toa = torch.Tensor(toa).to(self.device)

        if self.vis:
            video_id = str(data['ID'])[5:11]  # e.g.: b001_000490_*
            return features, labels, toa, detections, video_id
        else:
            return features, labels, toa


class CrashDataset(Dataset):
    def __init__(self, data_path, feature, phase='train', toTensor=False, device=torch.device('cuda'), vis=False):
        self.data_path = data_path
        self.feature = feature
        self.phase = phase
        self.toTensor = toTensor
        self.device = device
        self.vis = vis
        self.n_frames = 50
        self.n_obj = 19
        self.fps = 10.0
        self.dim_feature = self.get_feature_dim(feature)
        self.files_list, self.labels_list = self.read_datalist(data_path, phase)
        self.toa_dict = self.get_toa_all(data_path)

    def __len__(self):
        data_len = len(self.files_list)
        return data_len

    def get_feature_dim(self, feature_name):
        if feature_name == 'vgg16':
            return 4096
        elif feature_name == 'res101':
            return 2048
        else:
            raise ValueError

    def read_datalist(self, data_path, phase):
        # load training set
        list_file = os.path.join(data_path, self.feature + '_features', '%s.txt' % (phase))
        assert os.path.exists(list_file), "file not exists: %s"%(list_file)
        fid = open(list_file, 'r')
        data_files, data_labels = [], []
        for line in fid.readlines():
            filename, label = line.rstrip().split(' ')
            data_files.append(filename)
            data_labels.append(int(label))
        fid.close()
        return data_files, data_labels

    def get_toa_all(self, data_path):
        toa_dict = {}
        annofile = os.path.join(data_path, 'videos', 'Crash-1500.txt')
        annoData = self.read_anno_file(annofile)
        for anno in annoData:
            labels = np.array(anno['label'], dtype=np.int)
            toa = np.where(labels == 1)[0][0]
            toa = min(max(1, toa), self.n_frames-1)
            toa_dict[anno['vid']] = toa
        return toa_dict

    def read_anno_file(self, anno_file):
        assert os.path.exists(anno_file), "Annotation file does not exist! %s"%(anno_file)
        result = []
        with open(anno_file, 'r') as f:
            for line in f.readlines():
                items = {}
                items['vid'] = line.strip().split(',[')[0]
                labels = line.strip().split(',[')[1].split('],')[0]
                items['label'] = [int(val) for val in labels.split(',')]
                assert sum(items['label']) > 0, 'invalid accident annotation!'
                others = line.strip().split(',[')[1].split('],')[1].split(',')
                items['startframe'], items['vid_ytb'], items['lighting'], items['weather'], items['ego_involve'] = others
                result.append(items)
        f.close()
        return result

    def __getitem__(self, index):
        data_file = os.path.join(self.data_path, self.feature + '_features', self.files_list[index])
        assert os.path.exists(data_file), "file not exists: %s"%(data_file)
        try:
            data = np.load(data_file)
            features = data['data']  # 50 x 20 x 4096
            labels = data['labels']  # 2
            detections = data['det']  # 50 x 19 x 6
            vid = str(data['ID'])
        except:
            raise IOError('Load data error! File: %s'%(data_file))
        if labels[1] > 0:
            toa = [self.toa_dict[vid]]
        else:
            toa = [self.n_frames + 1]

        if self.toTensor:
            features = torch.Tensor(features).to(self.device)         #  50 x 20 x 4096
            labels = torch.Tensor(labels).to(self.device)
            toa = torch.Tensor(toa).to(self.device)

        if self.vis:
            return features, labels, toa, detections, vid
        else:
            return features, labels, toa


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import argparse
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data',
                        help='The relative path of dataset.')
    parser.add_argument('--dataset', type=str, default='dad', choices=['dad', 'crash'],
                        help='The name of dataset. Default: dad')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='The batch size in training process. Default: 10')
    parser.add_argument('--feature_name', type=str, default='vgg16', choices=['vgg16', 'res101'],
                        help='The name of feature embedding methods. Default: vgg16')
    p = parser.parse_args()

    seed = 123
    np.random.seed(seed)
    torch.manual_seed(seed)
    ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(ROOT_PATH, p.data_path, p.dataset)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # create data loader
    if p.dataset == 'dad':
        train_data = DADDataset(data_path, p.feature_name, 'training', toTensor=True, device=device)
        test_data = DADDataset(data_path, p.feature_name, 'testing', toTensor=True, device=device, vis=True)

    elif p.dataset == 'crash':
        train_data = CrashDataset(data_path, p.feature_name, 'train', toTensor=True, device=device)
        test_data = CrashDataset(data_path, p.feature_name, 'test', toTensor=True, device=device, vis=True)
    else:
        raise NotImplementedError
    traindata_loader = DataLoader(dataset=train_data, batch_size=p.batch_size, shuffle=True, drop_last=True)
    testdata_loader = DataLoader(dataset=test_data, batch_size=p.batch_size, shuffle=False, drop_last=True)

    for e in range(2):
        print('Epoch: %d'%(e))
        for i, (batch_xs, batch_ys, batch_toas) in tqdm(enumerate(traindata_loader), total=len(traindata_loader)):
            if i == 0:
                print('feature dim:', batch_xs.size())
                print('label dim:', batch_ys.size())
                print('time of accidents dim:', batch_toas.size())

    for e in range(2):
        print('Epoch: %d'%(e))
        for i, (batch_xs, batch_ys, batch_toas, detections, video_ids) in \
            tqdm(enumerate(testdata_loader), desc="batch progress", total=len(testdata_loader)):
            if i == 0:
                print('feature dim:', batch_xs.size())
                print('label dim:', batch_ys.size())
                print('time of accidents dim:', batch_toas.size())
