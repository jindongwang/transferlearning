#include <algorithm>
#include <vector>

#include "caffe/layers/gradient_scaler_layer.hpp"
#include "caffe/messenger.hpp"

namespace caffe {

class AdaptationCoefficientHandler: public Listener {
 public:
  AdaptationCoefficientHandler(float lower_bound, float upper_bound, 
                               float alpha, float max_iter, float* coeff)
      : lower_bound_(lower_bound), upper_bound_(upper_bound), alpha_(alpha),
        max_iter_(max_iter), coeff_(*coeff) {
    height_ = upper_bound_ - lower_bound_;
  }

  void handle(void* message) {
    int iter = *(static_cast<int*>(message));
    float progress = std::min(1.f, static_cast<float>(iter) / max_iter_);

    coeff_ = 2.f * height_ / (1.f + exp(-alpha_ * progress)) - 
             height_ + lower_bound_;

    // LOG(INFO) << "iter = " << iter << " progress = " << progress << " coeff = " << coeff_;
  }

 private:
  float lower_bound_, upper_bound_, alpha_, max_iter_, height_;
  float& coeff_;
};

template <typename Dtype>
void GradientScalerLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);

  CHECK_EQ(top.size(), bottom.size())
        << "top size should be the same with bottom size";
  
  lower_bound_ = this->layer_param_.gradient_scaler_param().lower_bound();
  upper_bound_ = this->layer_param_.gradient_scaler_param().upper_bound();
  threshold_iter_ = this->layer_param_.gradient_scaler_param().threshold_iter();
  iter_ = 0;
  alpha_ = this->layer_param_.gradient_scaler_param().alpha();
  max_iter_ = this->layer_param_.gradient_scaler_param().max_iter();
  coeff_ = 1.f; // Default adaptation coefficient.

  DCHECK(lower_bound_ <= upper_bound_);
  DCHECK(alpha_ >= 0.f);
  DCHECK(max_iter_ >= 1.f);
  
  Messenger::AddListener("SOLVER_ITER_CHANGED", 
      new AdaptationCoefficientHandler(lower_bound_, upper_bound_, 
                                       alpha_, max_iter_, &coeff_));
}

template <typename Dtype>
void GradientScalerLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    for(int i = 0;i < top.size();++i){
        top[i]->ReshapeLike(*bottom[i]);
    }
}

template <typename Dtype>
void GradientScalerLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  for(int i = 0;i < top.size();++i)
    top[i]->ShareData(*bottom[i]);
}

template <typename Dtype>
void GradientScalerLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for(int i = 0;i < top.size();++i){
      if (propagate_down[i]) {
        const int count = bottom[i]->count();
        const Dtype* top_diff = top[i]->cpu_diff();
        Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();

        caffe_cpu_scale(count, Dtype(-coeff_), top_diff, bottom_diff);
      }
  }
}

#ifdef CPU_ONLY
STUB_GPU(GradientScalerLayer);
#endif

INSTANTIATE_CLASS(GradientScalerLayer);
REGISTER_LAYER_CLASS(GradientScaler);

}  // namespace caffe
