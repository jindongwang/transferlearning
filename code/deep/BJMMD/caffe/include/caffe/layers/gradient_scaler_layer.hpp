#ifndef CAFFE_GRADIENT_SCALER_LAYER_HPP_
#define CAFFE_GRADIENT_SCALER_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"

namespace caffe {

template <typename Dtype>
class GradientScalerLayer : public NeuronLayer<Dtype> {
 public:
  /**
   * @param param provides GradientScalerParameter gradient_scaler_param,
   *     with GradientScalerLayer options:
   *   - lower_bound (\b optional, default 0) @f$ lower\_bound @f$
   *   - upper_bound (\b optional, default 1) @f$ upper\_bound @f$
   *   - alpha (\b optional, default 10) @f$ \alpha @f$
   *   - max_iter (\b optional, default 1) @f$ max\_iter @f$
   */
  explicit GradientScalerLayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "GradientScaler"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  virtual inline int ExactNumTopBlobs() const { return -1; }
  virtual inline int ExactNumBottomBlobs() const { return -1; }

  float lower_bound_, upper_bound_, alpha_, max_iter_, coeff_;
  int threshold_iter_, iter_;
};

}  // namespace caffe

#endif  // CAFFE_TANH_LAYER_HPP_
