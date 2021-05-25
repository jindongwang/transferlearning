# DAN
A PyTorch implementation of '[Learning Transferable Features with Deep Adaptation Networks](http://ise.thss.tsinghua.edu.cn/~mlong/doc/deep-adaptation-networks-icml15.pdf)'.
The contributions of this paper are summarized as follows. 
* They propose a novel deep neural network architecture for domain adaptation, in which all the layers corresponding to task-specific features are adapted in a layerwise manner, hence benefiting from “deep adaptation.”
* They explore multiple kernels for adapting deep representations, which substantially enhances adaptation effectiveness compared to single kernel methods. Our model can yield unbiased deep features with statistical guarantees.