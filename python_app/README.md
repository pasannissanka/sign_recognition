## Object Detection Benchmark

***

### YOLO Object detection OpenCV

#### Build instructions

* Install CUDA / cuDNN for nvidia gpu
* If CUDA compatible gpu isn't available, use CPU

*** 
To use CUDA on OpenCV, OpenCV needs to build from scratch and symlink to venv.
##### Instructions on how to build OpenCV 

* [How to use OpenCV’s “dnn” module with NVIDIA GPUs, CUDA, and cuDNN](https://pyimagesearch.com/2020/02/03/how-to-use-opencvs-dnn-module-with-nvidia-gpus-cuda-and-cudnn/)
* [Compiling and installing OpenCV DNN with CUDA
](https://gist.github.com/fengyuentau/28b72e4b83ee192434d66059a1ef00af)

*** 

* build script used
```shell
cmake -D CMAKE_BUILD_TYPE=RELEASE \
        -D CMAKE_INSTALL_PREFIX=/usr/local \
        -D INSTALL_PYTHON_EXAMPLES=ON \
        -D INSTALL_C_EXAMPLES=OFF \
        -D OPENCV_ENABLE_NONFREE=ON \
        -D WITH_CUDA=ON \
        -D WITH_CUDNN=ON \
        -D OPENCV_DNN_CUDA=ON \
        -D ENABLE_FAST_MATH=1 \
        -D CUDA_FAST_MATH=1 \
        -D CUDA_ARCH_BIN=8.6 \
        -D WITH_CUBLAS=1 \
        -D OPENCV_EXTRA_MODULES_PATH=/home/pasannissanka/Projects/SignRecognition/opencv/opencv_contrib/modules \
        -D HAVE_opencv_python3=ON \
        -D PYTHON_EXECUTABLE=~/.virtualenvs/opencv_cuda/bin/python \
        -D Eigen3_DIR=/home/pasannissanka/.local/share/eigen3/cmake \
        -D WITH_EIGEN=ON \
        -D BUILD_EXAMPLES=ON ..
```

