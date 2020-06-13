#####!## /usr/bin/env python3
"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2
import numpy as np
from random import randint

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60
# print(HOSTNAME, IPADDRESS, MQTT_HOST, 'HHHHHHHHHHHH')
def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client

def draw_masks(box, frame, width, height):
    xmin = int(box[3] * width)
    ymin = int(box[4] * height)
    xmax = int(box[5] * width)
    ymax = int(box[6] * height)
    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
    return frame

def count_people(result, frame, width, height, prob_threshold=0.1):
    frame = cv2.resize(frame.transpose((1, 2, 0)), (width, height), interpolation=cv2.INTER_NEAREST)
    total_in_frame = 0
    out_frame = frame
    frame_coor = []
    for box in result[0][0]:
        conf = box[2]
        class_idx = box[1]
        if conf >= prob_threshold and class_idx == 1:
            if frame_coor and frame_coor[0][2] > conf:
                frame_coor.append(box)
            else:
                total_in_frame += 1
                out_frame = draw_masks(box, frame, width, height)
                frame_coor.append(box)

    return out_frame, total_in_frame

def get_class_names(class_nums):
    class_names= []
    for i in class_nums:
        class_names.append(CLASSES[int(i)])
    return class_names

def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network(args)
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    ### TODO: Load the model through `infer_network` ###
    load_start_time = time.time()
    infer_network.load_model()
    time_to_load_model = time.time() - load_start_time

    ### TODO: Handle the input stream ###
    img_flag = False
    img_ext = {'jpg', 'jpeg', 'bmp', 'png'}
    if args.input == 'CAM':
        args.input = 0
    elif args.input.split('.')[1] in img_ext:
        img_flag = True

    input_shape = infer_network.get_input_shape()
    capture = cv2.VideoCapture(args.input)
    capture.open(args.input)

    width = int(capture.get(3))
    height = int(capture.get(4))
    
    # print(input_shape, width, height, 'FFFFFFFFFFFF')
    # cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    # 0x00000021
    if not img_flag:
        out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (width, height))
    else:
        out = None

    ### TODO: Loop until stream is over ###
    threshold_frames = 20
    total_people_counted = 0
    person_detected = False
    person_exit_frame = False
    start_exit_tracking = False
    start_time = None
    track_exit_time = 0
    count_exit_tracking_frames = 0
    total_frames_with_person = 0
    total_frames_with_person_detected = 0
    total_frames_with_person_not_detected = 0
    total_frames = 0
    total_inference_time = 0
    num_people_in_frame = 0
    while capture.isOpened():
        ### TODO: Read from the video capture ###
        flag, frame = capture.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)

        ### TODO: Pre-process the image as needed ###
        input_width = input_shape[3]
        input_height = input_shape[2]
        dim = (input_width, input_height)

        frame = cv2.resize(frame, dim)
        frame = frame.transpose((2,0,1))
        process_frame = frame.reshape(1,*frame.shape)

        ### TODO: Start asynchronous inference for specified request ###
        start_inference = time.time()
        infer_network.exec_net(process_frame)

        # track inference time and increment frames count
        total_inference_time += time.time() - start_inference
        total_frames += 1

        ### TODO: Wait for the result ###
        if infer_network.wait() == 0:
            ### TODO: Get the results of the inference request ###

            result = infer_network.get_output()
            ### TODO: Extract any desired stats from the results ###
            output_frame, persons_in_frame = count_people(result, frame, width, height, prob_threshold)

            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###        
            if persons_in_frame:
                if not person_detected:
                    count_exit_tracking_frames = 0
                    total_frames_with_person += 1
                    total_frames_with_person_detected += 1
                    start_time = time.time()
                    person_detected = True
                    num_people_in_frame = persons_in_frame
                else:
                    total_frames_with_person += 1
                    total_frames_with_person_detected += 1


                if start_exit_tracking:
                    count_exit_tracking_frames = 0
                    start_exit_tracking = False
            
            elif not persons_in_frame and person_detected:
                count_exit_tracking_frames += 1
                if count_exit_tracking_frames >= threshold_frames:
                    person_exit_frame = True
                    person_detected = False
                    start_exit_tracking = False
                    total_frames_with_person_not_detected = total_frames_with_person_not_detected - threshold_frames
                    total_frames_with_person += total_frames_with_person_not_detected
                    total_frames_with_person_not_detected = 0

                if not start_exit_tracking:
                    start_exit_tracking = True
                    track_exit_time = time.time()
                
                if not person_exit_frame:
                    total_frames_with_person_not_detected += 1

            if not persons_in_frame and person_exit_frame:
                    time_in_frame = track_exit_time - start_time
                    total_people_counted += num_people_in_frame
                    client.publish("person", json.dumps({ "count": num_people_in_frame, "total": total_people_counted }))
                    client.publish("person/duration", json.dumps({ "duration": time_in_frame }))
                    person_exit_frame = False

            if not img_flag:
                out.write(output_frame)




        ### TODO: Send the frame to the FFMPEG server ###
        # sys.stdout.buffer.write(output_frame)
        # sys.stdout.flush()

        ### TODO: Write an output image if `single_image_mode` ###
        if img_flag:
            cv2.imwrite('output_img.jpg', output_frame)

        if key_pressed == 27:
            break

    print(total_frames, 'TOTAL FRAME')
    print(total_inference_time/total_frames, 'INFERENCE_TIME')
    print(total_frames_with_person, 'TOTAL FRAMES WITH PERSON')
    print(total_frames_with_person_detected, 'TOAL FRAMES DETECTED')
    print(total_frames_with_person - total_frames_with_person_detected, 'DIFFERENCE IN FRAMES DETECTED/FALSE_POSITIVE')
    print(total_frames_with_person_not_detected, 'TOTAL FRAMES WITH PERSON NOT DETECTED')
    print(time_to_load_model, 'TIME TO LOAD MODEL')
    print(total_people_counted, 'TOTAL PEOPLE COUNTED')
    if not img_flag:
        out.release()
    capture.release()
    cv2.destroyAllWindows()
    client.disconnect()

def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)

if __name__ == '__main__':
    main()
