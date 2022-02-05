# Deep Feature Extractor

If we want to use some features extracted from deep networks such as ResNet, then this code will be of help.

## Supported Datasets

Currently, we support two kinds of datasets: `image` and `digit`. 
- Image datasets can be versatile.
- Digit datasets: we support MNIST, USPS, and SVHN.


## Requirements

Python 3, PyTorch 1.0+, PIL

## Usage

- For image dataset, go to folder `for_image_data`, then run:

`python main.py --dataset_path 'your_data_folder' --model_name resnet50 --src amazon --tar webcam`

- For digit dataset, go to folder `for_digit_data`, then run:

`python digit_deep_feature.py -src mnist -tar usps`

## Download Features that We Have Already Extracted

Currently, we support *ResNet-50* features since this architecture is very popular.

[Office-31 ResNet-50 features](https://pan.baidu.com/s/1UoyJSqoCKCda-NcP-zraVg)

[Office-Home ResNet-50 pretrained features](https://pan.baidu.com/s/1qvcWJCXVG8JkZnoM4BVoGg)

[Image-CLEF ResNet-50 pretrained features](https://pan.baidu.com/s/16wBgDJI6drA0oYq537h4FQ)

[VisDA classification dataset features by ResNet-50](https://pan.baidu.com/s/1sbuDqWWzwLyB1fFIpo5BdQ)

## Downloaded Finetuned Models

You can download finetuned models here:

Finetuned ResNet-50 models For Office-31 dataset: [BaiduYun](https://pan.baidu.com/s/1mRVDYOpeLz3siIId3tni6Q) | [Mega](https://mega.nz/#F!laI2lKoJ!nSmVQXrpu1Ov794sy2wFKg)

Finetuned ResNet-50 models For Office-Home dataset: [BaiduYun](https://pan.baidu.com/s/1i_g-QC2HZ0ZUhTnnySFIWw) | [Mega](https://mega.nz/#F!pGIkjIxC!MDD3ps6RzTXWobMfHh0Slw)

Finetuned ResNet-50 models For ImageCLEF dataset: [BaiduYun](https://pan.baidu.com/s/1y9tqyzBL7LZTd7Td380fxA) | [Mega](https://mega.nz/#F!QPJCzShS!b6qQUXWnCCGBMVs0m6MdQw)

Finetuned ResNet-50 models For VisDA dataset: [BaiduYun](https://pan.baidu.com/s/1DIcmmZ57ylMO6kpt46gkNQ) | [Mega](https://mega.nz/#F!ZDY2jShR!r_M2sR7MBi_9JPsRUXXy0g)

Finetuned LeNet+ models For MNIST dataset: [BaiduYun](https://pan.baidu.com/s/1W68JlO6z7BfYSo_OdMOpPg)

The names of the model on image datasets: `best_resnet_domain.pth`, while `domain` indicates the domain of the dataset.

The finetune procedure following a 8-2 training/validation split.

## Benchmark

See the power of deep features [here](https://github.com/jindongwang/transferlearning/blob/master/data/benchmark.md).