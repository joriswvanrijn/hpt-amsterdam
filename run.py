from models import *
from utils import *

import os, sys, time, random, argparse
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

import cv2
from sort import *

from csv import writer

def process_video(args):
    '''
    Actual video processing
    '''

    # Some config
    config_path = 'config/yolov3.cfg'
    weights_path = 'config/yolov3.weights'
    class_path = 'config/coco.names'
    img_size = 416
    colors=[(255,0,0),(0,255,0),(0,0,255),(255,0,255),(128,0,0),(0,128,0),(0,0,128),(128,0,128),(128,128,0),(0,128,128)]

    # Load model and weights
    model = Darknet(config_path, img_size=img_size)
    model.load_weights(weights_path)
    model.cuda()
    model.eval()
    classes = utils.load_classes(class_path)
    Tensor = torch.cuda.FloatTensor

    # Open video and init tracker
    cap = cv2.VideoCapture(args.input)
    mot_tracker = Sort() 

    # Live preview
    cv2.namedWindow('Stream', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Stream', (800,600))

    # First frame
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    retflag, frame = cap.read()
    vw = frame.shape[1]
    vh = frame.shape[0]

    # Set up output video
    output_path = args.input.replace(".mp4", "-detections.mp4")
    out = cv2.VideoWriter(output_path,fourcc,20.0,(vw,vh))

    # Set output csv
    csv_file = args.input.replace(".mp4", "-data.csv")
    prepare_csv(csv_file, args)

    # Loop over all the frames
    read_frames = 0
    start_time = datetime.now()

    while cap.isOpened():
        # Get the frame information
        retflag, frame = cap.read()
        read_frames += 1

        if retflag:
            # We are analysing one frame now
            print('Now considering frame:', read_frames)

            # Step 1: Detect on image
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pilimg = Image.fromarray(frame)
            detections = detect_image(args, pilimg, img_size, Tensor, model)

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            img = np.array(pilimg)
            pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
            pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
            unpad_h = img_size - pad_y
            unpad_w = img_size - pad_x

            # Step 2: Track detections
            if detections is not None:
                # Add objects to the tracker if we've found them
                tracked_objects = mot_tracker.update(detections.cpu())

                # Get the unique labels
                unique_labels = detections[:, -1].cpu().unique()

                n_cls_preds = len(unique_labels)
                for x1, y1, x2, y2, obj_id, cls_pred in tracked_objects:

                    # Step 3: For all unique detections we have found, draw a box
                    box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
                    box_w = int(((x2 - x1) / unpad_w) * img.shape[1])
                    y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
                    x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])
                    color = colors[int(obj_id) % len(colors)]
                    cls_label = classes[int(cls_pred)]
                    cv2.rectangle(frame, (x1, y1), (x1+box_w, y1+box_h), color, 4)
                    cv2.rectangle(frame, (x1, y1-35), (x1+len(cls_label)*19+80, y1), color, -1)
                    cv2.putText(frame, cls_label + "-" + str(int(obj_id)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1)

                    # Step 4: For all unique detections, save a line to the csv
                    with open(csv_file, 'a+', newline='') as write_obj:
                        csv_writer = writer(write_obj)
                        csv_writer.writerow([                   
                                            read_frames,        
                                            len(tracked_objects), 
                                            cls_label + "-" + str(int(obj_id)),   
                                            x1, y1, x2, y2
                                        ])

            # Add the image with boxes to the output video
            cv2.imshow('Stream', frame)
            out.write(frame)

            # Quit the loop if we close the window
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print('Analysis stopped by user')
                break

        else:
            # Quit the loop if no frames are left
            print('Analysis stopped because it ran out of frames')
            break

    # Shut down analaysis, we're done!
    end_time = datetime.now()

    print('Detection finished in %s' % (end_time - start_time))
    print('Total frames:', read_frames)

    # Save videos
    cap.release()
    out.release()

    # Close live preview
    cv2.destroyAllWindows()

    print('Video saved to ' + output_path)
    print('Data saved to ' + csv_file)

    return

def detect_image(args, img, img_size, Tensor, model):
    '''
    Get detections from frame
    '''

    # Scale and pad image
    ratio = min(img_size/img.size[0], img_size/img.size[1])
    imw = round(img.size[0] * ratio)
    imh = round(img.size[1] * ratio)
    img_transforms=transforms.Compose([transforms.Resize((imh,imw)),
        transforms.Pad((max(int((imh-imw)/2),0), 
            max(int((imw-imh)/2),0), max(int((imh-imw)/2),0),
            max(int((imw-imh)/2),0)), (128,128,128)),
        transforms.ToTensor(),
    ])

    # Convert image to Tensor
    image_tensor = img_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = Variable(image_tensor.type(Tensor))

    # Run inference on the model and get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = utils.non_max_suppression(detections, 80, args.obj_thresh, args.nms_thresh)

    return detections[0]

def prepare_csv(csv_file, args):
    '''
    Save a CSV for data tracking
    '''

    with open(csv_file, 'w', newline='') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow([
                                'Frame',      
                                'Total objects on frame',
                                'Object ID',
                                'TL x',
                                'TL y',
                                'BR x',
                                'BR y',
                            ])

    return

def parse_args():
    '''
    Pares argument
    '''

    parser = argparse.ArgumentParser(description='HPT-Amsterdam Video Analysis')
    parser.add_argument('-i', '--input', required=True, help='input video')
    parser.add_argument('-t', '--obj-thresh', type=float, default=0.8, help='objectness threshold, DEFAULT: 0.8')
    parser.add_argument('-n', '--nms-thresh', type=float, default=0.4, help='non max suppression threshold, DEFAULT: 0.4')
    args = parser.parse_args()

    return args

def main():
    '''
    Main routine
    '''
    args = parse_args()

    if not torch.cuda.is_available():
        print("ERROR: cuda is not available")
        sys.exit(1)

    assert args.input.endswith(
        '.mp4'), '{} is not a .mp4 file'.format(args.input)

    process_video(args)

    return

if __name__ == '__main__':
    main()