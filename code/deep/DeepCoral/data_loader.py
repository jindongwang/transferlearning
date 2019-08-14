from torchvision import datasets, transforms
import torch

def load_data(root_path, dir, batch_size, train, kwargs):
    transform = {
        'train': transforms.Compose(
            [transforms.Resize([256, 256]),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()]),
        'test': transforms.Compose(
            [transforms.Resize([224, 224]),
                transforms.ToTensor()])
        }
    data = datasets.ImageFolder(root = root_path + dir, transform=transform['train' if train else 'test'])
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, **kwargs, drop_last = True)
    return data_loader
 
