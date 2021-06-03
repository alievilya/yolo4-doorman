#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
import json

import warnings
from collections import OrderedDict
from os.path import join
from timeit import time

import cv2
import imutils.video
import numpy as np
import tensorflow as tf
from PIL import Image

from deep_sort import nn_matching
from deep_sort import preprocessing
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from rectangles import find_centroid, Rectangle, rect_square
from tools import generate_detections as gdet
from videocaptureasync import VideoCaptureAsync
from yolo import YOLO

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)

warnings.filterwarnings('ignore')
rect_endpoint_tmp = []
rect_bbox = []
bbox_list_rois = []
drawing = False


def find_ratio_ofbboxes(bbox, rect_compare):
    ratio = 0
    rect_detection = Rectangle(bbox[0], bbox[1],
                               bbox[2], bbox[3])
    inter_detection = rect_detection & rect_compare
    if inter_detection:
        inter_square_detection = rect_square(*inter_detection)
        cur_square_detection = rect_square(*rect_detection)
        try:
            ratio = inter_square_detection / cur_square_detection
        except ZeroDivisionError:
            ratio = 0
    return ratio


class Counter:
    def __init__(self, counter_in, counter_out, track_id):
        self.frames_without_moves = 0
        self.fps = 3
        self.people_init = OrderedDict()
        self.people_bbox = OrderedDict()
        self.cur_bbox = OrderedDict()
        self.rat_init = OrderedDict()
        # self.dissappeared_frames = OrderedDict()
        self.counter_in = counter_in
        self.counter_out = counter_out
        self.track_id = track_id

    def obj_initialized(self, track_id):
        self.people_init[track_id] = 0

    def clear_all(self):
        self.people_init = OrderedDict()
        self.people_bbox = OrderedDict()
        self.cur_bbox = OrderedDict()
        self.rat_init = OrderedDict()

    def need_to_clear(self):
        self.frames_without_moves += 1
        if self.frames_without_moves >= self.fps * 30 and len(self.people_init.keys()) > 0:
            self.frames_without_moves = 0
            return True

    def someone_inframe(self):
        self.frames_without_moves = 0

    def get_in(self):
        self.counter_in += 1

    def get_out(self):
        self.counter_out += 1

    def return_counter(self):
        return self.counter_in, self.counter_out

    def return_total_count(self):
        return self.counter_in + self.counter_out


def check_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Restrict TensorFlow to only use the fourth GPU
            # tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            tf.config.experimental.set_visible_devices(gpus, 'GPU')
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def main(yolo):
    # Definition of the parameters
    with open("cfg/detection_tracker_cfg.json") as detection_config:
        detect_config = json.load(detection_config)
    with open("data_files/doors_info.json") as doors_config:
        doors_config = json.load(doors_config)
    max_cosine_distance = 0.2
    nn_budget = None
    nms_max_overlap = 1.0
    output_format = '.mp4'
    input_format = '.mp4'


    model_filename = detect_config["tracking_model"]
    input_folder, output_folder = detect_config["input_folder"], detect_config["output_folder"]
    meta_folder = detect_config["meta_folder"]

    # Deep SORT
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    show_detections = True
    writeVideo_flag = True
    asyncVideo_flag = False

    fpeses = []

    check_gpu()
    video_name = 'test2'
    video_path = video_name + input_format
    print("opening video: {}".format(video_path))
    full_video_path = join(input_folder, video_path)
    # full_video_path = "rtsp://admin:admin@192.168.1.52:554/1/h264major"

    meta_name = meta_folder + video_name + ".json"
    with open(meta_name) as meta_config_json:
        meta_config = json.load(meta_config_json)
    camera_id = meta_config["camera_id"]
    if not os.path.exists(output_folder + str(camera_id)):
        os.mkdir(output_folder + str(camera_id))

    output_name = output_folder + camera_id + 'out_' + video_name + output_format
    counter = Counter(counter_in=0, counter_out=0, track_id=0)

    if asyncVideo_flag:
        video_capture = VideoCaptureAsync(full_video_path)
    else:
        video_capture = cv2.VideoCapture(full_video_path)

    if asyncVideo_flag:
        video_capture.start()
        w = int(video_capture.cap.get(3))
        h = int(video_capture.cap.get(4))
    else:
        w = int(video_capture.get(3))
        h = int(video_capture.get(4))

    if writeVideo_flag:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_name, fourcc, 25, (w, h))

    door_array = [0, 0, w / 2, h]
    fps = 0.0

    door_array = doors_config["{}".format(camera_id)]

    rect_door = Rectangle(door_array[0], door_array[1], door_array[2], door_array[3])

    border_door = door_array[3]
    while True:
        fps_imutils = imutils.video.FPS().start()
        ret, frame = video_capture.read()  # frame shape 640*480*3
        if not ret:
            with open('log_results.txt', 'a') as log:
                log.write('1')
            break

        t1 = time.time()

        image = Image.fromarray(frame[..., ::-1])  # bgr to rgb
        boxes, confidence, classes = yolo.detect_image(image)

        features = encoder(frame, boxes)
        detections = [Detection(bbox, confidence, cls, feature) for bbox, confidence, cls, feature in
                      zip(boxes, confidence, classes, features)]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.cls for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        cv2.rectangle(frame, (int(door_array[0]), int(door_array[1])), (int(door_array[2]), int(door_array[3])),
                      (23, 158, 21), 3)
        if len(detections) != 0:
            counter.someone_inframe()
            for det in detections:
                bbox = det.to_tlbr()
                if show_detections and len(classes) > 0:
                    score = "%.2f" % (det.confidence * 100) + "%"
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 3)
        else:
            if counter.need_to_clear():
                counter.clear_all()

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()

            if track.track_id not in counter.people_init or counter.people_init[track.track_id] == 0:
                counter.obj_initialized(track.track_id)
                ratio_init = find_ratio_ofbboxes(bbox=bbox, rect_compare=rect_door)

                if ratio_init > 0:
                    if ratio_init >= 0.5:  # and bbox[3] < door_array[3]:
                        counter.people_init[track.track_id] = 2  # man in the door
                    elif ratio_init < 0.5:  # and bbox[3] > door_array[3]:  # initialized in the outside
                        counter.people_init[track.track_id] = 1
                else:
                    counter.people_init[track.track_id] = 1
                counter.people_bbox[track.track_id] = bbox
            counter.cur_bbox[track.track_id] = bbox

            adc = "%.2f" % (track.adc * 100) + "%"  # Average detection confidence
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
            cv2.putText(frame, "ID: " + str(track.track_id), (int(bbox[0]), int(bbox[1]) + 50), 0,
                        1e-3 * frame.shape[0], (0, 255, 0), 5)

            if not show_detections:
                track_cls = track.cls
                cv2.putText(frame, str(track_cls), (int(bbox[0]), int(bbox[3])), 0, 1e-3 * frame.shape[0],
                            (0, 255, 0),
                            1)
                cv2.putText(frame, 'ADC: ' + adc, (int(bbox[0]), int(bbox[3] + 2e-2 * frame.shape[1])), 0,
                            1e-3 * frame.shape[0], (0, 255, 0), 1)

        id_get_lost = [track.track_id for track in tracker.tracks if track.time_since_update >= 35]

        # mb do clear people_init and other dicts
        for val in counter.people_init.keys():
            ratio = 0
            cur_c = find_centroid(counter.cur_bbox[val])
            init_c = find_centroid(counter.people_bbox[val])
            vector_person = (cur_c[0] - init_c[0],
                             cur_c[1] - init_c[1])

            if val in id_get_lost and counter.people_init[val] != -1:
                ratio = find_ratio_ofbboxes(bbox=counter.cur_bbox[val], rect_compare=rect_door)

                if vector_person[1] < -100 and counter.people_init[val] == 2 \
                        and ratio < 0.6:  # and counter.people_bbox[val][3] > border_door \
                    counter.get_out()
                    #  TODO save video in output_folder
                    print(vector_person[1], counter.people_init[val], ratio)

                elif vector_person[1] > 100 and counter.people_init[val] == 1 \
                        and ratio >= 0.6:
                    counter.get_in()
                    print(vector_person[1], counter.people_init[val], ratio)

                counter.people_init[val] = -1

        ins, outs = counter.return_counter()
        cv2.rectangle(frame, (700, 0), (950, 50),
                      (0, 0, 0), -1, 8)
        cv2.putText(frame, "in: {}, out: {} ".format(ins, outs), (710, 35), 0,
                    1e-3 * frame.shape[0], (255, 255, 255), 3)

        cv2.namedWindow('video', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('video', 1422, 800)
        cv2.imshow('video', frame)

        if writeVideo_flag:
            out.write(frame)

        fps_imutils.update()

        if not asyncVideo_flag:
            fps = (fps + (1. / (time.time() - t1))) / 2
            print("FPS = %f" % fps)

            if len(fpeses) < 15:
                fpeses.append(round(fps, 2))

            elif len(fpeses) == 15:
                # fps = round(np.median(np.array(fpeses)))
                median_fps = float(np.median(np.array(fpeses)))
                fps = round(median_fps, 1)
                print('max fps: ', fps)
                # fps = 20
                counter.fps = fps
                fpeses.append(fps)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if asyncVideo_flag:
        video_capture.stop()
    else:
        video_capture.release()

    if writeVideo_flag:
        out.release()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    tf.debugging.set_log_device_placement(True)
    gpus = tf.config.experimental.list_logical_devices('GPU')
    # with tf.device(gpus[1]):
    main(YOLO())
