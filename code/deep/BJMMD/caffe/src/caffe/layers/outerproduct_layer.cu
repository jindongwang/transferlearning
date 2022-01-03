#include <vector>

#include "caffe/layers/outerproduct_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void OuterProductForward(const int nthreads, const Dtype* data1, const Dtype* data2,
    const int axis1, const int axis2, Dtype* out_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
      int temp = axis1 * axis2;
     for(int i = 0; i < axis1; i++){
         for(int j = 0; j < axis2; j++){
             out_data[index * temp + i * axis2 + j] = data1[index * axis1 + i] * data2[index * axis2 + j];
         }
     }
  }
}

template <typename Dtype>
void OuterProductLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    const int nthreads = bottom[0]->shape(0);
    const int axis1 = bottom[0]->shape(1);
    const int axis2 = bottom[1]->shape(1);
    OuterProductForward<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
        nthreads, bottom[0]->gpu_data(), bottom[1]->gpu_data(), axis1, axis2, top[0]->mutable_gpu_data());
}

template <typename Dtype>
__global__ void OuterProductBackward(const int nthreads, const Dtype* data, const Dtype* data1, const Dtype* data2,
    const int axis1, const int axis2, Dtype* diff1, Dtype* diff2) {
  CUDA_KERNEL_LOOP(index, nthreads) {
      int temp = axis1 * axis2;
      for(int i = 0; i < axis1; i++){
           diff1[index * axis1 + i] = 0;
      }
    for(int j = 0; j < axis2; j++){
           diff2[index * axis2 + j] = 0;
      }
     for(int i = 0; i < axis1; i++){
         for(int j = 0; j < axis2; j++){
             diff1[index * axis1 + i] += data[index * temp + i * axis2 + j] * data2[index * axis2 + j];
         }
         //diff1[index * axis1 + i] /= Dtype(axis2);
     }
     for(int j = 0; j < axis2; j++){
         for(int i = 0; i < axis1; i++){
             diff2[index * axis2 + j] += data[index * temp + i * axis2 + j] * data1[index * axis1 + i];
         }
         //diff2[index * axis2 + j] /= Dtype(axis1);
     }
  }
}

template <typename Dtype>
void OuterProductLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    const int nthreads = bottom[0]->shape(0);
    const int axis1 = bottom[0]->shape(1);
    const int axis2 = bottom[1]->shape(1);
    OuterProductBackward<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
      nthreads, top[0]->gpu_diff(), bottom[0]->gpu_data(), bottom[1]->gpu_data(), axis1, axis2, bottom[0]->mutable_gpu_diff(), bottom[1]->mutable_gpu_diff());}
INSTANTIATE_LAYER_GPU_FUNCS(OuterProductLayer);

}  // namespace caffe
