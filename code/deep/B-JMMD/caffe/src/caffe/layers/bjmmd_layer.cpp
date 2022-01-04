#include <vector>

#include "caffe/layers/bjmmd_layer.hpp"

namespace caffe {

template <typename Dtype>
void BJMMDLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  dim_ = bottom[0]->count() / bottom[0]->count(0, 1);
  source_num_ = bottom[0]->count(0, 1);
  target_num_ = bottom[1]->count(0, 1);
  total_num_ = source_num_ + target_num_;
  kernel_num_ = this->layer_param_.bjmmd_param().kernel_num(); 
  label_kernel_num_ = this->layer_param_.bjmmd_param().label_kernel_num(); 
  fix_gamma_ = this->layer_param_.bjmmd_param().fix_gamma();
  sigma_ = this->layer_param_.bjmmd_param().sigma();
  auto_sigma_ = this->layer_param_.bjmmd_param().auto_sigma();
  gamma_ = Dtype(-1);
  kernel_mul_ = this->layer_param_.bjmmd_param().kernel_mul();
  label_kernel_mul_ = this->layer_param_.bjmmd_param().label_kernel_mul();
  balanced_factor_ = this->layer_param_.bjmmd_param().balanced_factor();
  diff_.Reshape(1, total_num_, total_num_, dim_);
  for(int i = 0;i < kernel_num_;++i){
      Blob<Dtype>* temp = new Blob<Dtype>(1, 1, total_num_, total_num_);
      kernel_val_.push_back(temp);
  }
  diff_multiplier_.Reshape(1, 1, 1, dim_);
  delta_.Reshape(1, 1, total_num_, total_num_);
  caffe_set(dim_, Dtype(1), diff_multiplier_.mutable_cpu_data());
  loss_weight_ = this->layer_param_.loss_weight(0);
  label_loss_weight_ = this->layer_param_.loss_weight(0);
}

template <typename Dtype>
void BJMMDLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  vector<int> loss_shape(0);
  top[0]->Reshape(loss_shape);
  if(top.size() == 2){
      top[1]->Reshape(loss_shape);
  }
  dim_ = bottom[0]->count() / bottom[0]->count(0, 1);
  source_num_ = bottom[0]->count(0, 1);
  target_num_ = bottom[1]->count(0, 1);
  total_num_ = source_num_ + target_num_;
  diff_.Reshape(1, total_num_, total_num_, dim_);
  for(int i = 0;i < kernel_num_;++i){
    kernel_val_[i]->Reshape(1, 1, total_num_, total_num_);
  }
  diff_multiplier_.Reshape(1, 1, 1, dim_);
  caffe_set(dim_, Dtype(1), diff_multiplier_.mutable_cpu_data());
  delta_.Reshape(1, 1, total_num_, total_num_);
}

template <typename Dtype>
void BJMMDLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
}
template <typename Dtype>
void BJMMDLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
}

#ifdef CPU_ONLY
STUB_GPU(BJMMDLossLayer);
#endif

INSTANTIATE_CLASS(BJMMDLossLayer);
REGISTER_LAYER_CLASS(BJMMDLoss);

}  // namespace caffe
