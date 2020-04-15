from models import *
from utils import *

import os, sys, time, datetime, random, argparse
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# Actual video processing

def process_video(args):
    config_path = 'config/yolov3.cfg'
    weights_path = 'config/yolov3.weights'
    class_path = 'config/coco.names'
    img_size = 416
    conf_thres = 0.8
    nms_thres = 0.4

    # Load model and weights
    model = Darknet(config_path, img_size=img_size)
    model.load_weights(weights_path)
    model.cuda()
    model.eval()
    
    classes = utils.load_classes(class_path)
    Tensor = torch.cuda.FloatTensor

# Main routine and argument parser

def parse_args():
    parser = argparse.ArgumentParser(description='HPT-Amsterdam Video Analysis')
    parser.add_argument('-i', '--input', required=True, help='input video')
    args = parser.parse_args()

    return args

def main():

    args = parse_args()

    if not torch.cuda.is_available():
        print("ERROR: cuda is not available")
        sys.exit(1)

    process_video(args)

    return

if __name__ == '__main__':
    main()