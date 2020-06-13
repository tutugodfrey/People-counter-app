import cv2
import numpy as np
import time
import sys
import tensorflow as tf
import pathlib

import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
# from matplotlib import pyplot as plt
from PIL import Image
from IPython.display import display

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

def load_model(model_name):
  model_dir = '/Users/godfreytutu/.keras/datasets/' + model_name
  model_dir = pathlib.Path(model_dir)/"saved_model"
  load_start_time = time.time()
  model = tf.saved_model.load(str(model_dir))
  time_to_load_model = time.time() - load_start_time
  model = model.signatures['serving_default']
  return model, time_to_load_model


def run_inference_for_single_image(model, image):
  person_detected = False
  image = np.asarray(image)
  input_tensor = tf.convert_to_tensor(image)

  input_tensor = input_tensor[tf.newaxis,...]
  start_of_inference = time.time()
  output_dict = model(input_tensor)
  inference_time = time.time() - start_of_inference

  num_detections = int(output_dict.pop('num_detections'))
  output_dict = {key: value[0, :num_detections].numpy() for key, value in output_dict.items()}
  output_dict['num_detections'] = num_detections
  output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
  
  if output_dict['detection_classes'].any():
    if tf.get_static_value(output_dict['detection_scores'][0]) >= 0.3 and output_dict['detection_classes'][0] == 1:
      person_detected = True


  # if model contains masks
  if 'detection_masks' in output_dict:
    detection_masks_reframe = utils_ops.reframe_box_masks_to_image_masks(
      output_dict['detection_masks'], output_dict['detection_boxes'], image.shape[0], image.shape[1])
    detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5, tf.uint8)
    output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

  return output_dict, inference_time, person_detected



def show_inference_video(model, frame):
  PATH_TO_LABELS = 'object_detection/data/mscoco_label_map.pbtxt'
  category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

  image_np = frame
  output_dict, inference_time, person_detected = run_inference_for_single_image(model, image_np)
  vis_util.visualize_boxes_and_labels_on_image_array(
    image_np,
    output_dict['detection_boxes'],
    output_dict['detection_classes'],
    output_dict['detection_scores'],
    category_index,
    instance_masks=output_dict.get('detection_masks_reframed', None),
    use_normalized_coordinates=True,
    line_thickness=8)

  return image_np, inference_time, person_detected


def make_inference():
  model_name = "ssd_mobilenet_v2_coco_2018_03_29"
  # model_name = "ssd_mobilenet_v1_coco_2017_11_17"
  # model_name = "faster_rcnn_nas_coco_24_10_2017"
  # model_name = 'ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03'
  # model_name = 'ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03'
  # model_name = 'ssd_inception_v2_coco_2018_01_28'
  detection_model, time_to_load_model = load_model(model_name)
  VIDEO_PATH='/Users/godfreytutu/Desktop/data-science/projects/people-counter/resources/Pedestrian_Detect_2_1_1.mp4'
  capture = cv2.VideoCapture(VIDEO_PATH)
  capture.open(VIDEO_PATH)
  width = int(capture.get(3))
  height = int(capture.get(4))

  # output = cv2.VideoWriter('output_ssd_2017.mp4', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (width, height))
  output = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (width, height))

  total_inference_time = 0
  total_frames_process = 0
  frames_threshold = 20
  person_in_frame = False
  frames_count = 0
  start_exit_tracking = False
  person_exit_frame = False

  total_frames_with_person = 0
  total_frames_with_person_detected = 0
  total_frames_with_person_not_detected = 0

  people_counted = 0
  counter2 = 0
  while capture.isOpened():
    flag, frame = capture.read()
    key_pressed = cv2.waitKey(60)
    if frame is  None:
      break

    output_frame, inference_time, person_detected = show_inference_video(detection_model, frame)
    total_inference_time += inference_time
    total_frames_process += 1
    output.write(output_frame)

    if person_detected:
      total_frames_with_person += 1
      total_frames_with_person_detected += 1
      frames_count = 0
      if not person_in_frame:
        counter2 += 1
        person_in_frame = True
        person_exit_frame = False

    elif not person_detected:
      if person_in_frame:
        frames_count += 1
      
      if not person_exit_frame:
        total_frames_with_person_not_detected += 1
      
      if frames_count >= frames_threshold:
        person_exit_frame = True
        person_in_frame = False
        frames_count = 0
        total_frames_with_person_not_detected = total_frames_with_person_not_detected - frames_threshold
        total_frames_with_person += total_frames_with_person_not_detected
        total_frames_with_person_not_detected = 0
        people_counted += 1

    if key_pressed == 27:
      break

  output.release()
  capture.release()
  cv2.destroyAllWindows()
  average_inference_time = total_inference_time / total_frames_process
  print(average_inference_time, 'TOTAL_INFERENCE_TIME')
  print(total_frames_process, 'TOTAL_FRAMES')
  print(time_to_load_model, 'TIME TO LOAD MODEL')
  print(total_frames_with_person, 'TOTAL FRAMES WITH PERSON')
  print(total_frames_with_person_detected, 'TOTAL FRAMES WITH PERSON DETECTED')
  print(total_frames_with_person - total_frames_with_person_detected, 'DIFFERENCE IN FRAMES DETECTED/FALSE_POSITIVE')
  print(people_counted, 'TOTAL PEOPLE COUNTED')

if __name__ == '__main__':
  make_inference()
