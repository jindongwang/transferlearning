# CLIP for transfer learning datasets

This demo shows you how to use OpenAI's [CLIP](https://openai.com/blog/clip/) model to perform *zero-shot* inference on existing transfer learning datasets.

## Requirements

First, make sure you have pytorch, torchvision, and CUDA installed. Then, install CLIP by the following commands:

```
$ pip install ftfy regex tqdm
$ pip install git+https://github.com/openai/CLIP.git
```

Then, you can install requirements by `pip install -r requirements.txt`.

## Usage

### Download data

At Microsoft, you can refer to `data/download_data_azcopy.py` to directly download the data you want from Azure blob. Super fast.

If you are outside Microsoft, you can use `data/download_data.py` to download datasets.

### How CLIP works

Before we start, we recommend that you understand the basics of CLIP. In a nutshell, CLIP is a large multi-muldality model trained by **text-image** pairs. Thus, CLIP model should take as inputs **both** a text and an image. 
Thus, if you want to classify an image, you should also pass the associated text to the model, e.g., a cat image and a text like "a photo of a cat".
That brings us the second difference of CLIP: it performs classification based on image-text similarity, rather than traditional linear layers for classification. So if you want to classify a cat image, you should pass at least two texts: "a photo of cat" and "a photo of dog".
Then, CLIP will compute the similarity between that image and the texts of cat and dog, then it returns the similarity score to both texts. Of course it will compute higher similarity score for cat text, which will finally classifiy the image as a cat.

### How to use this demo

The main file is `main.py`, which takes the following arguments:
- `model`, indicating which CLIP backbone (pre-trained) model you use.
- `dataset`, indicating which dataset you use.
- `gpu`, indicating which gpu you use.
- `mode`, which mode: 'zs' for zero-shot inference, 'ze' for feature extraction, and 'ft' for finetuning.
- `root`, the root folder for your datasets.
- `batchsize`, batchsize for training.

Other args can be set at will.


After downloading your datasets (read [here](https://github.com/jindongwang/transferlearning/tree/master/data)), you can run the script in the following style:

```
# run clip using ResNet-50 (model=0) as backbone on Art domain of Office-Home dataset (dataset=0)
python main.py --model 0 --dataset 0 --batchsize 512
```

Currently, this demo supports the following datasets: Office-Home, Office-31, PACS, VLCS, and ImageNet-R.
More datasets can be easily added according to your preference.

## Results

Again, note that the results are *zero-shot* by CLIP, which means, no domain adaptation, no fine-tuning, and no domain generalization, just simply run CLIP pre-trained models on the test domain to gather the results.
We also support finetune and domain-adaptation of CLIP but we did not run all results due to time limit.

### Office-Home

| backbone       | Art    | Clipart | Product | RealWorld | avg    |
|----------------|--------|---------|---------|-----------|--------|
| RN50           | 0.7268 | 0.5123  | 0.8238  | 0.8272    | 0.7225 |
| RN101          | 0.7684 | 0.5503  | 0.8468  | 0.8439    | 0.7524 |
| RN50x4         | 0.7956 | 0.5869  | 0.8723  | 0.8722    | 0.7818 |
| RN50x16        | 0.8220 | 0.6424  | 0.9018  | 0.8981    | 0.8161 |
| RN50x64        | 0.8686 | 0.7058  | 0.9315  | 0.9208    | 0.8567 |
| ViT-B-32       | 0.7804 | 0.6410  | 0.8756  | 0.8765    | 0.7934 |
| ViT-B-16       | 0.8278 | 0.6667  | 0.8950  | 0.8990    | 0.8221 |
| ViT-L-14       | 0.8669 | 0.7290  | 0.9299  | 0.9300    | 0.8640 |
| ViT-L-14@336px | 0.8838 | 0.7427  | 0.9399  | 0.9364    | 0.8757 |

### Office-31

| backbone       | amazon | webcam | dslr   | avg    |
|----------------|--------|--------|--------|--------|
| RN50           | 0.7249 | 0.6566 | 0.7430 | 0.6908 |
| RN101          | 0.7391 | 0.7585 | 0.7610 | 0.7488 |
| RN50x4         | 0.7636 | 0.8038 | 0.7912 | 0.7837 |
| RN50x16        | 0.7508 | 0.8340 | 0.8153 | 0.7924 |
| RN50x64        | 0.8204 | 0.8943 | 0.8454 | 0.8574 |
| ViT-B-32       | 0.7767 | 0.8101 | 0.8112 | 0.7934 |
| ViT-B-16       | 0.7969 | 0.8038 | 0.7972 | 0.8004 |
| ViT-L-14       | 0.8161 | 0.8352 | 0.8675 | 0.8257 |
| ViT-L-14@336px | 0.8229 | 0.8151 | 0.8394 | 0.8190 |

### PACS

| backbone       | A      | C      | P      | S      | avg    |
|----------------|--------|--------|--------|--------|--------|
| RN50           | 0.9229 | 0.9518 | 0.9946 | 0.8045 | 0.9184 |
| RN101          | 0.9463 | 0.9761 | 0.9946 | 0.8801 | 0.9493 |
| RN50x4         | 0.9336 | 0.9765 | 0.9647 | 0.8246 | 0.9249 |
| RN50x16        | 0.9507 | 0.9846 | 0.9976 | 0.8979 | 0.9577 |
| RN50x64        | 0.9648 | 0.9825 | 1.0000 | 0.9147 | 0.9655 |
| ViT-B-32       | 0.9585 | 0.9765 | 0.9970 | 0.8547 | 0.9467 |
| ViT-B-16       | 0.9746 | 0.9910 | 0.9994 | 0.8880 | 0.9633 |
| ViT-L-14       | 0.9883 | 0.9902 | 0.9994 | 0.9478 | 0.9814 |
| ViT-L-14@336px | 0.9888 | 0.9915 | 0.9994 | 0.9552 | 0.9837 |

### VLCS

| backbone       | C      | L      | S      | V      | avg    |
|----------------|--------|--------|--------|--------|--------|
| RN50           | 0.9894 | 0.6849 | 0.7182 | 0.8400 | 0.8081 |
| RN101          | 0.9972 | 0.5693 | 0.6584 | 0.7275 | 0.7381 |
| RN50x4         | 0.9640 | 0.6160 | 0.7072 | 0.7029 | 0.7475 |
| RN50x16        | 0.9965 | 0.5606 | 0.7157 | 0.7764 | 0.7623 |
| RN50x64        | 0.9993 | 0.5222 | 0.6840 | 0.8433 | 0.7622 |
| ViT-B_32       | 0.9993 | 0.6702 | 0.7154 | 0.8477 | 0.8082 |
| ViT-B_16       | 0.9993 | 0.6766 | 0.7508 | 0.8288 | 0.8139 |
| ViT-L_14       | 0.9993 | 0.6950 | 0.7035 | 0.8243 | 0.8055 |
| ViT-L_14@336px | 0.9993 | 0.6453 | 0.7179 | 0.8409 | 0.8009 |

### DomainNet

| backbone       | Clipart | Infograph | Painting | Quickdraw | Real   | Sketch | avg    |
|----------------|---------|-----------|----------|-----------|--------|--------|--------|
| RN50           | 0.5158  | 0.3920    | 0.5281   | 0.0627    | 0.7688 | 0.4886 | 0.4593 |
| RN101          | 0.5981  | 0.4070    | 0.5676   | 0.1030    | 0.7935 | 0.5417 | 0.5018 |
| RN50x4         | 0.6335  | 0.461     | 0.6131   | 0.1001    | 0.8115 | 0.5799 | 0.5332 |
| RN50x16        | 0.6876  | 0.4715    | 0.6351   | 0.1266    | 0.8232 | 0.6301 | 0.5624 |
| RN50x64        | 0.7328  | 0.5024    | 0.6763   | 0.1626    | 0.8463 | 0.6749 | 0.5992 |
| ViT-B-32       | 0.6703  | 0.3992    | 0.6239   | 0.1318    | 0.8054 | 0.5853 | 0.5360 |
| ViT-B-16       | 0.7091  | 0.4679    | 0.6599   | 0.1442    | 0.8315 | 0.6343 | 0.5745 |
| ViT-L-14       | 0.7795  | 0.4958    | 0.6913   | 0.2247    | 0.8599 | 0.7023 | 0.6256 |
| ViT-L-14@336px | 0.7860  | 0.5226    | 0.7078   | 0.2231    | 0.8662 | 0.7163 | 0.6370 |

### TerraInc

| backbone       | L38    | L43    | L46    | L100   | avg    |
|----------------|--------|--------|--------|--------|--------|
| RN50           | 0.1361 | 0.3297 | 0.2169 | 0.0884 | 0.1928 |
| RN101          | 0.4197 | 0.3748 | 0.2674 | 0.2474 | 0.3273 |
| RN50x4         | 0.2626 | 0.3567 | 0.2438 | 0.3164 | 0.2949 |
| RN50x16        | 0.3532 | 0.4715 | 0.3427 | 0.3626 | 0.3825 |
| RN50x64        | 0.4083 | 0.4990 | 0.3672 | 0.5817 | 0.4641 |
| ViT-B-32       | 0.1339 | 0.3071 | 0.1844 | 0.1346 | 0.1900 |
| ViT-B-16       | 0.1958 | 0.3350 | 0.3165 | 0.5117 | 0.3398 |
| ViT-L-14       | 0.4008 | 0.4597 | 0.3760 | 0.5182 | 0.4387 |
| ViT-L-14@336px | 0.4295 | 0.4892 | 0.4071 | 0.5100 | 0.4590 |

### VisDA-17 and ImageNet-R

| backbone       | VisDA-validation | ImageNet-R |
|----------------|------------------|------------|
| RN50           | 0.8049           | 0.5622     |
| RN101          | 0.8261           | 0.6239     |
| RN50x4         | 0.8219           | 0.6695     |
| RN50x16        | 0.8439           | 0.7477     |
| RN50x64        | 0.8569           | 0.8003     |
| ViT-B-32       | 0.8424           | 0.6667     |
| ViT-B-16       | 0.8633           | 0.7360     |
| ViT-L-14       | 0.8594           | 0.8474     |
| ViT-L-14@336px | 0.8628           | 0.8604     |


### Acknowledgements

- OpenAI's CLIP code: https://github.com/openai/CLIP
- CLIP paper: Radford A, Kim J W, Hallacy C, et al. Learning transferable visual models from natural language supervision[C]//International Conference on Machine Learning. PMLR, 2021: 8748-8763.