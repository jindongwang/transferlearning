#include <algorithm>
#include <vector>

#include "caffe/layers/outerproduct_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void OuterProductLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    vector<int> temp_shape;
    temp_shape.push_back(bottom[0]->shape(0));
    temp_shape.push_back(1);
    temp_shape.push_back(bottom[0]->shape(1));
    temp_shape.push_back(bottom[1]->shape(1));
    top[0]->Reshape(temp_shape);
}

template <typename Dtype>
void OuterProductLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    vector<int> temp_shape;
    temp_shape.push_back(bottom[0]->shape(0));
    temp_shape.push_back(1);
    temp_shape.push_back(bottom[0]->shape(1));
    temp_shape.push_back(bottom[1]->shape(1));
    top[0]->Reshape(temp_shape);
}

template <typename Dtype>
void OuterProductLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void OuterProductLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
}

#ifdef CPU_ONLY
STUB_GPU(OuterProductLayer);
#endif

INSTANTIATE_CLASS(OuterProductLayer);
REGISTER_LAYER_CLASS(OuterProduct);

}  // namespace caffe
