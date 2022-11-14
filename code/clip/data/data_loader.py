import torchvision.datasets as datasets
from torchvision import transforms
import os
import clip

from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageTextData(object):

    def __init__(self, dataset, root, preprocess, prompt='a picture of a'):
        if type(dataset) is int:
            dataset = self._DATA_FOLDER[dataset]
        dataset = os.path.join(root, dataset)
        if dataset == 'imagenet-r':
            data = datasets.ImageFolder(
                'imagenet-r', transform=self._TRANSFORM)
            labels = open('imagenetr_labels.txt').read().splitlines()
            labels = [x.split(',')[1].strip() for x in labels]
        else:
            data = datasets.ImageFolder(dataset, transform=self._TRANSFORM)
            labels = data.classes
        self.data = data
        self.labels = labels
        if prompt:
            self.labels = [prompt + ' ' + x for x in self.labels]

        self.preprocess = preprocess
        self.text = clip.tokenize(self.labels)

    def __getitem__(self, index):
        image, label = self.data.imgs[index]
        if self.preprocess is not None:
            image = self.preprocess(Image.open(image))
        text_enc = self.text[label]
        return image, text_enc, label

    def __len__(self):
        return len(self.data)

    @staticmethod
    def get_data_name_by_index(index):
        name = ImageTextData._DATA_FOLDER[index]
        name = name.replace('/', '_')
        return name

    _DATA_FOLDER = [
        'dataset/OfficeHome/Art',
        'dataset/OfficeHome/Clipart',
        'dataset/OfficeHome/Product',
        'dataset/OfficeHome/RealWorld',

        'dataset/office31/amazon',   # 4
        'dataset/office31/webcam',
        'dataset/office31/dslr',

        'dataset/VLCS/Caltech101',   # 7
        'dataset/VLCS/LabelMe',
        'dataset/VLCS/SUN09',
        'dataset/VLCS/VOC2007',

        'dataset/PACS/kfold/art_painting',   # 11
        'dataset/PACS/kfold/cartoon',
        'dataset/PACS/kfold/photo',
        'dataset/PACS/kfold/sketch',

        'dataset/visda/validation',  # 15

        'dataset/domainnet/clipart',  # 16
        'dataset/domainnet/infograph',
        'dataset/domainnet/painting',
        'dataset/domainnet/quickdraw',
        'dataset/domainnet/real',
        'dataset/domainnet/sketch',

        'dataset/terra_incognita/location_38',  # 22
        'dataset/terra_incognita/location_43',
        'dataset/terra_incognita/location_46',
        'dataset/terra_incognita/location_100',

        'imagenet-r',
    ]

    _TRANSFORM = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


if __name__ == '__main__':
    print(ImageTextData.get_data_name_by_index(0))
