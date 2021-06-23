# -*- coding: utf-8 -*-
"""
References/Adapted From:
https://www.pyimagesearch.com/2015/05/25/basic-motion-detection-and-tracking-with-python-and-opencv/

Description:
This script runs a motion detector! It detects transient motion in a room
and said movement is large enough, and recent enough, reports that there is
motion!

Run the script with a working webcam! You'll see how it works!
"""

import json
import socket
import threading
import time
from time import gmtime, strftime

import cv2

from rectangles import Rectangle


class MoveDetector():
    def __init__(self, id, sock):
        self.sock = sock
        self.camera_id = id
        # self.link = link
        self.link = "rtsp://admin:admin@192.168.1.{}:554/1/h264major".format(self.camera_id)
        # Init frame variables
        self.around_door_array = [None, None, None, None]
        self.cap = None
        self.ret = None
        self.first_frame = None
        self.next_frame = None
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.delay_counter = 0
        self.movement_persistent_counter = 0
        self.fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        self.fps = 20
        self.output_video = None
        self.contours = []
        self.videos_meta_folder = "data_files/videos_meta/"
        self.meta_file = {}
        self.video_name = None

    def load_doors(self):
        with open("cfg/around_doors_info.json") as doors_config:
            self.around_doors_config = json.load(doors_config)
        self.around_door_array = self.around_doors_config[self.camera_id]
        self.rect_around_door = Rectangle(self.around_door_array[0], self.around_door_array[1],
                                          self.around_door_array[2], self.around_door_array[3])

    def read_cap(self):
        if self.cap.isOpened():
            self.ret, self.frame = self.cap.read()

    def set_init(self):
        self.read_cap()
        self.load_doors()
        self.transient_movement_flag = False
        self.stop_writing = False
        # Read frame
        self.text = "Unoccupied"

    def start_video(self, frame, output_name):
        if self.output_video is None:
            self.output_video = cv2.VideoWriter(output_name, self.fourcc, self.fps, (frame.shape[1], frame.shape[0]))
            print('started')

    def write_video(self):
        if self.output_video:
            if self.output_video.isOpened():
                self.output_video.write(self.frame)
                print('writing')

    def gen_meta(self):
        self.meta_file["camera_id"] = self.camera_id
        self.meta_file["link"] = self.link

    def write_meta(self):
        self.gen_meta()
        with open(self.videos_meta_folder + self.video_name[:-4] + '.json', 'w') as wr:
            json.dump(self.meta_file, wr)

    def send_socket(self):
        self.sock.sendall(bytes(self.video_name + ';', "utf-8"))
        # data = sock.recv(100)
        # print('Received: ', repr(data.decode("utf-8")))

    def release_video(self, frame):
        if self.output_video:
            # if self.output_video.isOpened():
            self.output_video.write(frame)
            self.output_video.release()
            self.write_meta()
            print('released')
            self.stop_writing = False
            self.output_video = None
            if self.sock:
                self.send_socket()

    def move_near_door(self, contours):
        if len(contours) > 0:
            return True
        else:
            return False

    def detect_movement(self, config):
        if not self.ret:
            print("CAPTURE ERROR")
            # continue
            return False
        # Resize and save a greyscale version of the image

        # self.frame = imutils.resize(self.frame, width=640)
        self.cropped_frame = self.frame[self.around_door_array[1]:self.around_door_array[3],
                             self.around_door_array[0]:self.around_door_array[2]]
        self.gray = cv2.cvtColor(self.cropped_frame, cv2.COLOR_BGR2GRAY)
        # Blur it to remove camera noise (reducing false positives)
        self.gray = cv2.GaussianBlur(self.gray, (17, 17), 0)
        # If the first frame is nothing, initialise it
        if self.first_frame is None: self.first_frame = self.gray
        self.delay_counter += 1
        # Otherwise, set the first frame to compare as the previous frame
        # But only if the counter reaches the appriopriate value
        # The delay is to allow relatively slow motions to be counted as large
        # motions if they're spread out far enough
        if self.delay_counter > config["FRAMES_TO_PERSIST"]:
            delay_counter = 0
            self.first_frame = self.next_frame
        # Set the next frame to compare (the current frame)
        self.next_frame = self.gray
        # Compare the two frames, find the difference
        self.frame_delta = cv2.absdiff(self.first_frame, self.next_frame)
        thresh = cv2.threshold(self.frame_delta, 20, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=4)
        cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # loop over the contours
        self.contours = []
        for c in cnts:
            # Save the coordinates of all found contours
            (x, y, w, h) = cv2.boundingRect(c)
            # If the contour is too small, ignore it, otherwise, there's transient
            # movement
            if cv2.contourArea(c) > config["MIN_SIZE_FOR_MOVEMENT"]:
                self.transient_movement_flag = True
                # Draw a rectangle around big enough movements
                top_coords = (x, y)
                bot_coords = (x + w, y + h)
                if config["draw_rect"] == "True":
                    cv2.rectangle(self.frame, top_coords, bot_coords, (0, 255, 0), 2)
                coords = top_coords + bot_coords
                self.contours.append(coords)

        # The moment something moves momentarily, reset the persistent
        # movement timer.
        if self.transient_movement_flag == True:
            self.movement_persistent_flag = True
            self.movement_persistent_counter = config["MOVEMENT_DETECTED_PERSISTENCE"]

        if self.movement_persistent_counter > 0:
            text = "Movement Detected " + str(self.movement_persistent_counter)
            self.movement_persistent_counter -= 1
        else:
            text = "No Movement Detected"
            self.stop_writing = True
            return self.frame, []
        if config["draw_rect"] == "True":
            cv2.putText(self.frame, str(text), (10, 35), self.font, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

        # self.frame_delta = cv2.cvtColor(self.frame_delta, cv2.COLOR_GRAY2BGR)

    def loop_detection(self):
        # print('opening link: ', self.link)

        while True:
            t0 = time.time()
            # print('opening link: ', self.link)
            if self.cap is None:
                self.cap = cv2.VideoCapture(self.link)  # start the cam
                # self.cap = VideoCaptureAsync(self.link)
            self.set_init()
            # print(self.frame.shape)
            if not self.ret:
                print('error: ret is None')
                # continue
                break
            self.detect_movement(config=config)

            if self.move_near_door(self.contours):
                if not self.video_name:
                    hour_greenwich = strftime("%H", gmtime())
                    # f'{self.link}_' +
                    hour_moscow = str(int(hour_greenwich) + 3)
                    self.video_name = hour_moscow + strftime("_%M_%S", gmtime()) + '.mp4'
                output_name = 'data_files/videos_motion/' + self.video_name
                self.start_video(self.frame, output_name)

            self.write_video()
            if self.stop_writing:
                self.release_video(self.frame)
                self.video_name = None

            # delta_time = (time.time() - t0)
            # fps = round(1 / delta_time)
            # print('fps = ', fps)


HOST = "localhost"
PORT = 8075

if __name__ == "__main__":
    with open("cfg/motion_detection_cfg.json") as config_file:
        config = json.load(config_file)
    ids = ['54', '52']

    fpeses = []

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        Motion = [MoveDetector(id_, sock) for id_ in ids]
        sock.connect((HOST, PORT))

        while True:
            t0 = time.time()
            channels = [threading.Thread(target=Mot.loop_detection, daemon=True) for Mot in Motion]
            for i, ch in enumerate(channels):
                # ch.start()
                if not ch.is_alive():
                    print(f"{Motion[i].link} \n")
                    ch.start()
                else:
                    print(f"{Motion[i].link} is alive")
            for ch in channels:
                ch.join()

            # delta_time = (time.time() - t0)
            # fps = round(1 / delta_time)
            # print('fps = ', fps)
            # if len(fpeses) < 35:
            #     fpeses.append(fps)
            #     print(delta_time)
            # elif len(fpeses) == 35:
            #     # fps = round(np.median(np.array(fpeses)))
            #     median_fps = float(np.median(np.array(fpeses)))
            #     fps = round(median_fps, 2)
            #     print('fps set: ', fps)
            #     fpeses.append(fps)

    # Cleanup when closed
    # cv2.destroyAllWindows()
    # cap.release()
