import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset


class AEDataset(Dataset):

    def __init__(self, data_path, mode, train_ratio, test_ratio, valid_ratio, labeled_ratio, unlabeled_ratio, slide_win, slide_stride, scale, random_seed=42):
        super().__init__()
        '''
        random split
        1 splitting the data through the windows and strides
        2 randomly choosing the indices and split train,test,valid 
        '''
        self.mode = mode
        self.w = slide_win
        self.s = slide_stride
        self.random_seed = random_seed

        xlist, ylist = [], []

        for file_name in os.listdir(data_path):
            data = torch.tensor(pd.read_csv(os.path.join(data_path, file_name)).values, dtype=torch.float32)
            features, labels = data[:, :-1], data[:, -1]

            features = self.scale_data(features, scale)

            x_, y_ = self.process(features, labels)
            xlist.append(x_)
            ylist.append(y_)

        self.x = torch.concat(xlist)
        self.y = torch.concat(ylist)

        self.random_split(train_ratio, test_ratio, valid_ratio, labeled_ratio, unlabeled_ratio)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        feature = self.x[idx]
        ground_truth = self.y[idx]

        return feature, ground_truth

    def scale_data(self, features, scale):
        if scale == 1:  # Standardization
            mean = features.mean(0)
            std = features.std(0)
            return (features - mean) / (std + 1e-7)  # Avoid division by zero with small epsilon

        elif scale == 2:  # Absolute Scaling
            max_abs = torch.max(torch.abs(features), dim=0)[0]
            return features / (max_abs + 1e-7)  # Avoid division by zero with small epsilon

        elif scale == 3:  # Min-Max Scaling
            feature_min = features.min(0)[0]
            feature_max = features.max(0)[0]
            range_ = feature_max - feature_min
            return (features - feature_min) / (range_ + 1e-7)  # Avoid division by zero when min == max

        else:
            return features  # No scaling if scale is not 1, 2, or 3

    def process(self, data, labels):
        x_list, y_list = [], []

        for i in range(self.w, len(data), self.s):
            ft = data[i - self.w:i, :]
            targ = labels[i]
            x_list.append(ft)
            y_list.append(targ)

        x = torch.stack(x_list).contiguous()
        y = torch.stack(y_list).contiguous()

        return x, y

    def random_split(self, train_ratio, test_ratio, valid_ratio, labeled_ratio, unlabeled_ratio):
        total_num = len(self.x)
        indices = torch.randperm(total_num, generator=torch.manual_seed(self.random_seed))

        train_num = int(train_ratio * total_num)
        labeled_num = int(labeled_ratio * train_num)
        unlabeled_num = int(unlabeled_ratio * train_num) if unlabeled_ratio < 1 else train_num - labeled_num
        test_num = int(test_ratio * total_num)
        valid_num = int(valid_ratio * total_num)

        train_indices = indices[:train_num]
        train_labeled_indices = train_indices[:labeled_num]
        train_unlabeled_indices = train_indices[labeled_num:labeled_num + unlabeled_num]
        test_indices = indices[train_num:train_num + test_num]
        valid_indices = indices[train_num + test_num:train_num + test_num + valid_num]

        splits = {
            'train': train_indices,
            'train_labeled': train_labeled_indices,
            'train_unlabeled': train_unlabeled_indices,
            'valid': valid_indices,
            'test': test_indices
        }

        if self.mode in splits:
            selected_indices = splits[self.mode]
            self.x = self.x[selected_indices]
            self.y = self.y[selected_indices]
        else:
            raise ValueError("Invalid mode. Choose from 'train', 'valid', or 'test'.")
