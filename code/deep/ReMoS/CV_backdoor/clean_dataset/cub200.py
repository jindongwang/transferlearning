import torch.utils.data as data
from PIL import Image
import random
import time
import numpy as np
import os

class CUB200Data(data.Dataset):
    def __init__(self, root, is_train=True, transform=None, shots=-1, seed=0, preload=False):
        self.num_classes = 200
        self.transform = transform
        self.preload = preload
        mapfile = os.path.join(root, 'images.txt') 
        imgset_desc = os.path.join(root, 'train_test_split.txt')
        labelfile = os.path.join(root, 'image_class_labels.txt')

        assert os.path.exists(mapfile), 'Mapping txt is missing ({})'.format(mapfile)
        assert os.path.exists(imgset_desc), 'Split txt is missing ({})'.format(imgset_desc)
        assert os.path.exists(labelfile), 'Label txt is missing ({})'.format(labelfile)

        self.img_ids = []
        max_id = 0
        with open(imgset_desc) as f:
            for line in f:
                i = int(line.split(' ')[0])
                s = int(line.split(' ')[1].strip())
                if s == is_train:
                    self.img_ids.append(i)
                if max_id < i:
                    max_id = i

        self.id_to_path = {}
        with open(mapfile) as f:
            for line in f:
                i = int(line.split(' ')[0])
                path = line.split(' ')[1].strip()
                self.id_to_path[i] = os.path.join(root, 'images', path)

        self.id_to_label = -1*np.ones(max_id+1, dtype=np.int64) # ID starts from 1
        with open(labelfile) as f:
            for line in f:
                i = int(line.split(' ')[0])
                #NOTE: In the network, class start from 0 instead of 1
                c = int(line.split(' ')[1].strip())-1
                self.id_to_label[i] = c

        if is_train:
            self.img_ids = np.array(self.img_ids)
            new_img_ids = []
            for c in range(self.num_classes):
                ids = np.where(self.id_to_label == c)[0]
                random.seed(seed)
                random.shuffle(ids)
                count = 0
                for i in ids:
                    if i in self.img_ids:
                        new_img_ids.append(i)
                        count += 1
                    if count == shots:
                        break
            self.img_ids = np.array(new_img_ids)

        self.imgs = {}
        if preload:
            for idx, id in enumerate(self.img_ids):
                if idx % 100 == 0:
                    print('Loading {}/{}...'.format(idx+1, len(self.img_ids)))
                img = Image.open(self.id_to_path[id]).convert('RGB')
                self.imgs[id] = img

    def __getitem__(self, index):
        img_id = self.img_ids[index]
        img_label = self.id_to_label[img_id]

        if self.preload:
            img = self.imgs[img_id]
        else:
            img = Image.open(self.id_to_path[img_id]).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        
        return img, img_label

    def __len__(self):
        return len(self.img_ids)
     

if __name__ == '__main__':
    seed= int(time.time())
    data_train = CUB200Data('/data/CUB_200_2011', True, shots=10, seed=seed)
    print(len(data_train))
    data_test = CUB200Data('/data/CUB_200_2011', False, shots=10, seed=seed)
    print(len(data_test))
    for i in data_train.img_ids:
        if i in data_test.img_ids:
            print('Test in training...')
    print('Test PASS!')
    print('Train', data_train.img_ids[:5])
    print('Test', data_test.img_ids[:5])
