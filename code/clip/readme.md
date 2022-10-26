# CLIP for transfer learning datasets

This demo shows you how to use OpenAI's [CLIP](https://openai.com/blog/clip/) model to perform *zero-shot* inference on existing transfer learning datasets.

## Requirements

First, make sure you have pytorch, torchvision, and CUDA installed. Then, install CLIP by the following commands:

```
$ pip install ftfy regex tqdm
$ pip install git+https://github.com/openai/CLIP.git
```

## Usage

### How CLIP works

Before we start, we recommend that you understand the basics of CLIP. In a nutshell, CLIP is a large multi-muldality model trained by **text-image** pairs. Thus, CLIP model should take as inputs **both** a text and an image. 
Thus, if you want to classify an image, you should also pass the associated text to the model, e.g., a cat image and a text like "a photo of a cat".
That brings us the second difference of CLIP: it performs classification based on image-text similarity, rather than traditional linear layers for classification. So if you want to classify a cat image, you should pass at least two texts: "a photo of cat" and "a photo of dog".
Then, CLIP will compute the similarity between that image and the texts of cat and dog, then it returns the similarity score to both texts. Of course it will compute higher similarity score for cat text, which will finally classifiy the image as a cat.

### How to use this demo

The main file is `test_clip.py`, which takes two arguments:
- `model`, indicating which CLIP backbone (pre-trained) model you use;
- `dataset`, indicating which dataset you use.

After downloading your datasets (read [here](https://github.com/jindongwang/transferlearning/tree/master/data)), you can run the script in the following style:

```
# run clip using ResNet-50 (model=0) as backbone on Art domain of Office-Home dataset (dataset=0)
python test_clip.py --model 0 --dataset 0  
```

Currently, this demo supports the following datasets: Office-Home, Office-31, PACS, VLCS, and ImageNet-R.
More datasets can be easily added according to your preference.

## Results

Again, note that the results are *zero-shot* by CLIP, which means, no domain adaptation, no fine-tuning, and no domain generalization, just simply run CLIP pre-trained models on the test domain to gather the results.

### Office-Home


### Office-31


### VLCS


### PACS


### ImageNet-R


### Acknowledgements

- OpenAI's CLIP code: https://github.com/openai/CLIP
- CLIP paper: Radford A, Kim J W, Hallacy C, et al. Learning transferable visual models from natural language supervision[C]//International Conference on Machine Learning. PMLR, 2021: 8748-8763.