from torchvision import datasets, transforms
import torch

def load_training(root_path, dir, batch_size):
    transform = transforms.Compose(
        [transforms.Resize([227,227])
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
         ])
    data = datasets.ImageFolder(root=root_path + dir, transform=transform)
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True,num_workers=4)
    return train_loader

def load_testing(root_path, dir, batch_size):
    transform = transforms.Compose(
        [transforms.Resize([227, 227]),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
         ])
    data = datasets.ImageFolder(root=root_path + dir, transform=transform)
    test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False,num_workers=4)
    return test_loader