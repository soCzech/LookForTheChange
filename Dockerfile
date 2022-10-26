FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

RUN apt-get update \
 && apt-get install ffmpeg libsm6 libxext6 libopenblas-dev -y

RUN pip install \
        opencv-python \
        pillow \
        matplotlib \
        scikit-learn \
        scipy \
        tqdm \
        pandas \
        ffmpeg-python \
        mxnet-cu110==1.8 \
        tensorflow-gpu==2.4 \
        tensorflow_hub

ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

RUN cd /usr/local/cuda/lib64 \
 && ln -s libcusolver.so.10 libcusolver.so.11

COPY cuda_ops /tmp

RUN cd /tmp \
 && TORCH_CUDA_ARCH_LIST="6.1;7.0;7.5;8.0" python setup.py install \
 && rm -rf *
