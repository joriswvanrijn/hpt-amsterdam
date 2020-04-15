# HPT Amsterdam

Python project to detect and track objects in HPT video's for HPT Amsterdam research.

## Methodology

**1. YOLOv3 Object detection**
- Pytorch implementation by Erik Lindernoren: https://github.com/eriklindernoren/PyTorch-YOLOv3
- Paper: https://pjreddie.com/media/files/papers/YOLOv3.pdf

**2. SORT**
- Implementation by Alex Bewley: https://github.com/abewley/sort
- Paper: https://arxiv.org/pdf/1602.00763.pdf

_Method by Chris Fotache: https://github.com/cfotache/pytorch_objectdetecttrack_

## Usage

```
python run.py -i demo/traffic.mp4
```

> Only .mp4 are accepted

Output files will be found in the data folder: `{video}-detections.mp4` and `{video}-data.csv`

In the /demo folder you'll find demo CSV output from the traffic video (data/traffic.mp4). A corresponding output video: https://youtu.be/eG22gWuhfD8

## Pratical limitations

- In streets with cars parked on the shoulders of the road, ID's may be mixed up/recreated due to a close proximity
- When two detected objects of the same class (e.g. two 'cars') pass behind/in front of eachother, ID's may be switched
- In this trained model, large cars are sometimes considered 'trucks', somtimes 'cars'

## Installation

### Requirements

- Ubuntu 18.04 LTS
- Python 3.5.6
- CUDA 10.1
- cuDNN 7.6.5
- TensorRT 5.1.2.2 RC for CUDA 10.0
- NVIDIA SMI Driver: 440.64.00

```
# Add NVIDIA package repositories
# Add HTTPS support for apt-key
sudo apt-get install gnupg-curl
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_10.1.243-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1604_10.1.243-1_amd64.deb
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
sudo apt-get update
wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb
sudo apt install ./nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb
sudo apt-get update

# Install NVIDIA driver
# Issue with driver install requires creating /usr/lib/nvidia
sudo mkdir /usr/lib/nvidia
sudo apt-get install --no-install-recommends nvidia-440
# Reboot. Check that GPUs are visible using the command: nvidia-smi

# Install development and runtime libraries (~4GB)
sudo apt-get install --no-install-recommends \
    cuda=10.0.130-1 \
    libcudnn7=7.6.5.32-1+cuda10.0  \
    libcudnn7-dev=7.6.5.32-1+cuda10.0

# Manually download cuDNN: https://developer.nvidia.com/rdp/cudnn-download
# copy the files to the cuda installation folder (usually /usr/loca/cuda/)

# Manually download TensorRT from: https://developer.nvidia.com/nvidia-tensorrt-5x-download
dpkg -i nv-tensorrt-repo-ubuntu1804-cuda10.0-trt5.1.2.2-rc-20190227_1-1_amd64.deb
apt-key add /var/nv-tensorrt-repo-cuda10.0-trt5.1.2.2-rc-20190227/7fa2af80.pub
apt-get update
apt-get install libnvinfer5=5.1.2-1+cuda10.0
apt-get install libnvinfer-dev=5.1.2-1+cuda10.0
apt-mark hold libnvinfer5 libnvinfer-dev
```

### Installing python dependencies

```
pip install tensorflow-gpu==1.14
pip install opencv-python==4.2.0.34
pip install keras==2.3.1
pip install numpy
pip install -U scikit-learn
pip install scikit-image
pip install numba==0.47.0
pip install filterpy==1.4.5
```

### Before first run

Download yolov3 weights to /config (https://pjreddie.com/media/files/yolov3.weights)

### Author & references

April 15, 2020

Joris W. van Rijn (hi@joriswvanrijn.com)

#### References:

- YOLOv3 Pytorch implementation by Erik Lindernoren: https://github.com/eriklindernoren/PyTorch-YOLOv3
- YOLO: https://pjreddie.com/media/files/papers/YOLOv3.pdf
- Sort Implementation by Alex Bewley: https://github.com/abewley/sort
- SORT: https://arxiv.org/pdf/1602.00763.pdf
