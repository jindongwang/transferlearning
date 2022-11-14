import numpy as np
from torchvision import transforms, datasets
from PIL import Image
import torch
import argparse
import clip
import os

CLIP_MODELS = [
    'RN50',
    'RN101',
    'RN50x4',
    'RN50x16',
    'RN50x64',
    'ViT-B/32',
    'ViT-B/16',
    'ViT-L/14',
    'ViT-L/14@336px'
]

DATA_FOLDER = [
    'OfficeHome/Art',
    'OfficeHome/Clipart',
    'OfficeHome/Product',
    'OfficeHome/RealWorld',

    'OFFICE31/amazon',
    'OFFICE31/webcam',
    'OFFICE31/dslr',

    'VLCS/VLCS/Caltech101',
    'VLCS/VLCS/LabelMe',
    'VLCS/VLCS/SUN09',
    'VLCS/VLCS/VOC2007',

    'PACS/kfold/art_painting',
    'PACS/kfold/cartoon',
    'PACS/kfold/photo',
    'PACS/kfold/sketch',

    'imagenet-r',
]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=int, default=5, help='model index')
    parser.add_argument('--dataset', type=int, default=0, help='dataset name')
    args = parser.parse_args()
    return args


def load_data(dataset):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    if dataset == 'imagenet-r':
        imagenet_r = datasets.ImageFolder('imagenet-r', transform=transform)
        imagenetr_labels = open('imagenetr_labels.txt').read().splitlines()
        imagenetr_labels = [x.split(',')[1].strip() for x in imagenetr_labels]
        return imagenet_r, imagenetr_labels
    else:
        officehome = datasets.ImageFolder(dataset, transform=transform)
        officehome_labels = officehome.classes
        return officehome, officehome_labels


def load_model(modelname):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(modelname, device)
    return model, preprocess


def classify_imagenetr(imagenet_r, imagenetr_labels, model, preprocess, device):
    res = []

    for item in imagenet_r.imgs:
        img, label = item
        image = Image.open(img)
        image_input = preprocess(image).unsqueeze(0).to(device)
        text_inputs = torch.cat(
            [clip.tokenize(f"a picture of a {c}") for c in imagenetr_labels]).to(device)

        # Calculate features
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_inputs)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, indices = similarity[0].topk(1)

        for value, index in zip(values, indices):
            res.append([index.cpu().numpy(), label])
    res = np.array(res)
    acc = np.mean(np.array(res)[:, 0] == np.array(res)[:, 1])
    return res, acc

def perform_inference(model_index, data_index):
    model_pretrain, dataset = CLIP_MODELS[model_index], DATA_FOLDER[data_index]
    data, labels = load_data(dataset)
    model, processor = load_model(model_pretrain)
    res, acc = classify_imagenetr(
        data, labels, model, processor, device='cuda')
    m_rep, d_rep = model_pretrain.replace('/', '-'), dataset.replace('/', '-')
    # if exist some folder
    if not os.path.exists('clip_res/'):
        os.makedirs('clip_res/')
    fname = f'clip_res/{m_rep}_{d_rep}'
    np.savetxt(fname + '.txt', res, fmt='%d')
    with open(fname + '.txt', 'w') as fp:
        fp.write(fname + ',' + str(acc))
    return res, acc

def gather_res(mid):
    model_name = CLIP_MODELS[mid]
    import glob
    files = glob.glob(f'clip_res/*{model_name}*.txt')
    new_f = f'res_all_{mid}.txt'
    with open(new_f, 'w') as f_n:
        for f in files:
            with open(f, 'r') as fp:
                s = fp.read()
                f_n.write(s + '\n')

def test():
    args = get_args()
    imagenet_r, imagenetr_labels = load_data(DATA_FOLDER[args.dataset])
    model, processor = load_model(CLIP_MODELS[args.model])
    res, acc = classify_imagenetr(
        imagenet_r, imagenetr_labels, model, processor, device='cuda')
    res = np.array(res)
    np.savetxt('res.txt', res, fmt='%d')
    print(acc)

def sweep():
    # Gives all results from all datasets across all models
    for mid in range(len(CLIP_MODELS)):
        for did in range(len(DATA_FOLDER)):
            print(CLIP_MODELS[mid], DATA_FOLDER[did])
            _, acc = perform_inference(mid, did)
            print(f'{acc:.2f}')
        gather_res(mid)


if __name__ == '__main__':
    test()
    
    
