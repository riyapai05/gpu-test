# GPU Image Classification

A simple PyTorch image classification project that runs inside Docker
using an NVIDIA GPU.

## Technologies

- Python
- PyTorch
- Docker
- CUDA
- NVIDIA GPU

## Run

Build the Docker image:

docker build -t gpu-image-classifier .

Run using NVIDIA GPU:

docker run --rm --gpus all gpu-image-classifier