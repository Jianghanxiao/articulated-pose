#!/usr/bin/env bash
CUDA_DIR=/usr/local/cuda-11.0/
CONDA_ENV_DIR=/localhome/hja40/miniconda3/envs/ANCSH
nvcc_bin=$CUDA_DIR/bin/nvcc

cuda_include_dir=$CUDA_DIR/include
tensorflow_include_dir=$CONDA_ENV_DIR/lib/python3.6/site-packages/tensorflow/include
tensorflow_external_dir=$CONDA_ENV_DIR/lib/python3.6/site-packages/tensorflow/include/external/nsync/public

cuda_library_dir=$CUDA_DIR/lib64/
tensorflow_library_dir=$CONDA_ENV_DIR/lib/python3.6/site-packages/tensorflow

