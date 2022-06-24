import torch.utils.data as data
from PIL import Image
import glob
import time
import numpy as np
import random
import os


class Stanford40Data(data.Dataset):
    def __init__(self, root, is_train=False, transform=None, shots=-1, seed=0, preload=False):
        self.num_classes = 40
        self.transform = transform
        first_line = True
        self.cls_names = []
        with open(os.path.join(root, 'ImageSplits', 'actions.txt')) as f:
            for line in f:
                if first_line:
                    first_line = False
                    continue
                self.cls_names.append(line.split('\t')[0].strip())

        if is_train:
            post = 'train'
        else:
            post = 'test'

        self.labels = []
        self.image_path = []

        for label, cls_name in enumerate(self.cls_names):
            with open(os.path.join(root, 'ImageSplits', '{}_{}.txt'.format(cls_name, post))) as f:
                for line in f:
                    self.labels.append(label)
                    self.image_path.append(os.path.join(root, 'JPEGImages', line.strip()))


        if is_train:
            self.labels = np.array(self.labels)
            new_image_path = []
            new_labels = []
            for c in range(self.num_classes):
                ids = np.where(self.labels == c)[0]
                random.seed(seed)
                random.shuffle(ids)
                count = 0
                for i in ids:
                    new_image_path.append(self.image_path[i])
                    new_labels.append(self.labels[i])
                    count += 1
                    if count == shots:
                        break
            self.labels = new_labels
            self.image_path = new_image_path

        self.imgs = []
        if preload:
            for idx, p in enumerate(self.image_path):
                if idx % 100 == 0:
                    print('Loading {}/{}...'.format(idx+1, len(self.image_path)))
                self.imgs.append(Image.open(p).convert('RGB'))

    def __getitem__(self, index):
        if len(self.imgs) > 0:
            img = self.imgs[index]
        else:
            img = Image.open(self.image_path[index]).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        
        return img, self.labels[index]

    def __len__(self):
        return len(self.labels)
     
if __name__ == '__main__':
    seed= int(98)
    data_train = Stanford40Data('/data/stanford_40', True, shots=10, seed=seed)
    print(len(data_train))
    data_test = Stanford40Data('/data/stanford_40', False, shots=10, seed=seed)
    print(len(data_test))
    for i in data_train.image_path:
        if i in data_test.image_path:
            print('Test in training...')
    print('Test PASS!')
    print('Train', data_train.image_path[:5])
    print('Test', data_test.image_path[:5])
