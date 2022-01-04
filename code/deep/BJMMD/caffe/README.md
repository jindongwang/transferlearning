# Deep Transfer Learning on Caffe

This is a caffe library for deep transfer learning. We fork the repository with version ID `29cdee7` from [Caffe](https://github.com/BVLC/caffe) and [Xlearn](https://github.com/thuml/Xlearn), and make our modifications. The main modifications is listed as follow:
- Add `bjmmd layer` described in paper "Balanced joint maximum mean discrepancy for deep transfer learning" (AA '2020).


Data Preparation
---------------
In `data/office/*.txt`, we give the lists of three domains in [Office](https://cs.stanford.edu/~jhoffman/domainadapt/#datasets_code) dataset.

We have published the Image-Clef dataset we use [here](https://drive.google.com/file/d/0B9kJH0-rJ2uRS3JILThaQXJhQlk/view?usp=sharing).

Training Model
---------------

In `models/B-JMMD/alexnet`, we give an example model based on Alexnet to show how to transfer from `amazon` to `webcam`. In this model, we insert bjmmd layers after fc7 and fc8 individually.In `models/models/B-JMMD/resnet`, we give an example model based on ResNet to show how to transfer from `amazon` to `webcam`.

The [bvlc\_reference\_caffenet](http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel) is used as the pre-trained model for Alexnet. The [deep-residual-networks](https://github.com/KaimingHe/deep-residual-networks) is used as the pre-trained model for Resnet. We use Resnet-50. If the Office dataset and pre-trained caffemodel are prepared, the example can be run with the following command:
```
Alexnet:

"./build/tools/caffe train -solver models/B-JMMD/alexnet/solver.prototxt -weights models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel"
```
```
ResNet:

"./build/tools/caffe train -solver models/B-JMMD/resnet/solver.prototxt -weights models/deep-residual-networks/ResNet-50-model.caffemodel"
```

Parameter Tuning
---------------
In bjmmd-layer, parameter `loss_weight` can be tuned to give jmmd loss different weights, parameter `balanced_factor` α can be tuned to give bjmmd loss a different marginal distribution daptation weights(α) or conditional distribution daptation weights(1-α) repectively.(balanced_factor α ∈ [0,1])

Changing Transfer Task
---------------
If you want to change to other transfer tasks (e.g. `webcam` to `amazon`), you may need to:

- In `train_val.prototxt` please change the source and target datasets;
- In `solver.prototxt` please change `test_iter` to the size of the target dataset: `2817` for `amazon`, `795` for `webcam` and `498` for `dslr`;
- you may also need to tune the `loss_weight` λ  and `balanced_factor`α to achieve the best accuracy, For each transfer task, you can first set the balance factor α to 0.5(JMMD), and tried your best to search the loss weight λ by changing it from 0 to 1 by a progressive schedulefor, for instance,`loss_weight` λ within {0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 1.0}, to achieve better accuracy, then fixed the loss weight λ obtained in the previous step and searched for the balance factor α within {0, 0.1, . . . , 0.9, 1.0} to achieve the best results.

## Citation
If you use this library for your research, we would be pleased if you cite the following papers:
```
    @article{Chuangji2020Balanced,
        title={Balanced Joint Maximum Mean Discrepancy for Deep Transfer Learning},
        author={Chuangji Meng and Cunlu Xu and Qin Lei and Wei Su and Jinzhao Wu},
        journal={Analysis and Applications},
        number={2},
        year={2020},
    }
        
```
   
