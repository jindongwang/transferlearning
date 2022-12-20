import clip
from PIL import Image
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from utils import convert_models_to_fp32


class ClipModel(object):

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

    def __init__(self, model_name='Vit-B/32', device='cuda', logger=None):
        self.device = device
        self.logger = logger
        if type(model_name) is int:
            model_name = self.index_to_model(model_name)
        self.model, self.preprocess = clip.load(
            model_name, device=device, jit=False)
        self.model.eval()
        self.model.to(device)
        self.model_name = model_name

    def index_to_model(self, index):
        return self.CLIP_MODELS[index]

    @staticmethod
    def get_model_name_by_index(index):
        name = ClipModel.CLIP_MODELS[index]
        name = name.replace('/', '_')
        return name

    def get_image_features(self, image, need_preprocess=False):
        if need_preprocess:
            image = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image)
        return image_features

    def get_text_feature(self, text):
        text = clip.tokenize(text).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text)
        return text_features

    def get_text_features_list(self, texts, train=False):
        if train:
            text_inputs = torch.cat([clip.tokenize(c)
                                    for c in texts]).to(self.device)
            text_features = self.model.encode_text(text_inputs)
        else:
            with torch.no_grad():
                text_inputs = torch.cat([clip.tokenize(c)
                                         for c in texts]).to(self.device)
                text_features = self.model.encode_text(text_inputs)

        return text_features

    def get_similarity(self, image_features, text_features):
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        return similarity

    def get_topk(self, image, text, k=1):
        similarity = self.get_similarity(image, text)
        values, indices = similarity[0].topk(k)
        return values, indices

    def feature_extraction(self, dataloader):
        res = None
        for batch in tqdm(dataloader):
            image, _, label = batch
            image = image.to(self.device)
            label = label.to(self.device)
            image_features = self.get_image_features(image)
            feat_lab = torch.cat(
                [image_features, label.view(-1, 1)], dim=1)
            if res is None:
                res = torch.zeros((1, feat_lab.shape[1])).to(self.device)
            res = torch.cat([res, feat_lab], dim=0)
        res = res[1:, :].cpu().numpy()
        return res


    def finetune(self, dataloader, testloader, optimizer, nepochs=10, save_path=None):
        loss_img = nn.CrossEntropyLoss()
        loss_txt = nn.CrossEntropyLoss()
        best_acc = 0
        for epoch in range(nepochs):
            total_loss = 0
            for batch in tqdm(dataloader):
                optimizer.zero_grad()
                image, text, _ = batch
                image = image.to(self.device)
                text = text.to(self.device)
                logits_per_image, logits_per_text = self.model(image, text)

                ground_truth = torch.arange(
                    len(image), dtype=torch.long, device=self.device)

                loss = (loss_img(logits_per_image, ground_truth) +
                        loss_txt(logits_per_text, ground_truth))/2
                loss.backward()
                total_loss += loss.item()
                if self.device == "cpu":
                    optimizer.step()
                else:
                    convert_models_to_fp32(self.model)
                    optimizer.step()
                    clip.model.convert_weights(self.model)

            eval_acc, _ = self.evaluate(testloader)
            if eval_acc > best_acc:
                best_acc = eval_acc
                if save_path is not None:
                    torch.save(self.model.state_dict(), save_path)
            self.logger.info("Epoch {} : Loss {}, Acc {:.4f}".format(
                epoch, total_loss/len(dataloader), eval_acc))
        return best_acc


    def evaluate(self, dataloader, modelpath=None):
        if modelpath is not None:
            self.model.load_state_dict(torch.load(modelpath))
        texts = dataloader.dataset.labels
        text_features = self.get_text_features_list(texts)
        res = None
        for batch in tqdm(dataloader):
            image, _, label = batch
            image = image.to(self.device)
            label = label.to(self.device)
            image_features = self.get_image_features(image)
            similarity = self.get_similarity(image_features, text_features)
            _, indices = similarity.topk(1)

            pred = torch.squeeze(indices)
            result = torch.cat([pred.view(-1, 1), label.view(-1, 1)], dim=1)
            if res is None:
                res = result
            else:
                res = torch.cat([res, result], dim=0)
        res = res.cpu().numpy()
        acc = np.mean(np.array(res)[:, 0] == np.array(res)[:, 1])
        return acc, res


if __name__ == '__main__':
    print(ClipModel.CLIP_MODELS)
    model_name = 'ViT-B/32'
    model_name = 5
    device = 'cuda'
    clip_inference = CLIP_INFERENCE(model_name, device)  # type: ignore
    print(clip_inference.model_name)

    image = Image.open('../test.jpg')
    text = 'a picture of a cat'
    print(clip_inference.inference(image, text))  # type: ignore

    dataset = datasets.ImageFolder(
        root='../dataset/office31/amazon',
        transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            ),
        ])
    )
    labels = dataset.classes
    res, acc = clip_inference.classification(dataset, labels)
    print(res)
    print(acc)
