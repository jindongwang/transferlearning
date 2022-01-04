#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/entropy_loss_layer.hpp"
namespace caffe {

template <typename Dtype>
void EntropyLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
    data_num_ = bottom[0]->count(0,1); 
    label_num_ = bottom[0]->count(1);
    ignore_label_ = this->layer_param_.entropy_param().ignore_label();
    threshold_ = this->layer_param_.entropy_param().threshold();
    loss_weight_ = this->layer_param_.loss_weight(0);
    iterations_num_ = this->layer_param_.entropy_param().iterations_num();
    now_iteration_ = 0;
}

template <typename Dtype>
void EntropyLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  normalized_bottom_data_.Reshape(1, 1, data_num_, label_num_);
}

template <typename Dtype>
void EntropyLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void EntropyLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

}

#ifdef CPU_ONLY
STUB_GPU(EntropyLossLayer);
#endif

INSTANTIATE_CLASS(EntropyLossLayer);
REGISTER_LAYER_CLASS(EntropyLoss);

}  // namespace caffe
