cd src
protoc ./caffe/proto/caffe.proto --cpp_out=../include/
cd ..
make -j48
