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
import time
from time import gmtime, strftime

import cv2
import numpy as np

import imutils

from rectangles import Rectangle, find_ratio_ofbboxes


class MoveDetector():
    def __init__(self):
        # Init frame variables
        self.around_door_array = [234, 45, 281, 174]
        self.rect_around_door = Rectangle(self.around_door_array[0], self.around_door_array[1],
                                          self.around_door_array[2], self.around_door_array[3])
        self.first_frame = None
        self.next_frame = None
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.delay_counter = 0
        self.movement_persistent_counter = 0
        self.fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        self.fps = 20
        self.output_video = None

    def set_init(self, cap):
        self.transient_movement_flag = False
        self.stop_writing = False
        # Read frame
        self.ret, self.frame = cap.read()
        self.text = "Unoccupied"
        return self.ret

    def start_video(self, frame, output_name):
        if self.output_video is None:
            self.output_video = cv2.VideoWriter(output_name, self.fourcc, self.fps, (frame.shape[1], frame.shape[0]))
            print('started')

    def write_video(self, frame):
        if self.output_video:
            if self.output_video.isOpened():
                print('writing')
                self.output_video.write(frame)

    def release_video(self, frame):
        if self.output_video:
            if self.output_video.isOpened():
                self.output_video.write(frame)
                self.output_video.release()
                print('released')
                self.stop_writing = False
                self.output_video = None

    def move_near_door(self, contours):
        if len(contours) > 0:
            for contour in contours:
                # bbox = contour[0] + contour[1]
                ratio_contour = find_ratio_ofbboxes(contour, rect_compare=self.rect_around_door)
                if ratio_contour >= 0:
                    return True
                else:
                    return False
        else:
            return False

    def detect_movement(self, config):
        if not self.ret:
            print("CAPTURE ERROR")
            # continue
            return False
        # Resize and save a greyscale version of the image
        self.frame = imutils.resize(self.frame, width=1280)
        self.gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
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
        moving_cnts = []
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
                moving_cnts.append(coords)

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

        cv2.putText(self.frame, str(text), (10, 35), self.font, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

        self.frame_delta = cv2.cvtColor(self.frame_delta, cv2.COLOR_GRAY2BGR)
        return self.frame, moving_cnts


if __name__ == "__main__":
    with open("cfg/motion_detection_cfg.json") as config_file:
        config = json.load(config_file)

    Motion = [MoveDetector() for _ in range(1)]
    link ="rtsp://admin:admin@192.168.1.52:554/1/h264major"
    # link = config["source"]
    print('opening link: ', link)
    cap = cv2.VideoCapture(link)  # Then start the webcam
    ret = True
    # LOOP!
    fpeses = []
    while True:
        t0 = time.time()
        for i in range(len(Motion)):

            ret = Motion[i].set_init(cap=cap)
            if not ret:
                break
            frame, contours = Motion[i].detect_movement(config=config)

            cv2.imshow("frame {}".format(i), frame)
            cv2.waitKey(1)
            if Motion[i].move_near_door(contours):
                hour_greenvich = strftime("%H", gmtime())
                hour_moscow = f'{i}_' + str(int(hour_greenvich) + 3)
                video_name = hour_moscow + strftime("_%M_%S", gmtime()) + '.mp4'
                output_name = 'data_files/videos_motion/' + video_name
                Motion[i].start_video(frame, output_name)
            Motion[i].write_video(frame)
            if Motion[i].stop_writing:
                Motion[i].release_video(frame)

            # Interrupt trigger by pressing q to quit the open CV program
            # ch = cv2.waitKey(3)
            # if ch & 0xFF == ord('q'):
            #     break
        delta_time = (time.time() - t0)
        # if len(fpeses) < 35:
        #     fpeses.append(round(1 / delta_time))
        #     print(delta_time)
        # elif len(fpeses) == 35:
        #     # fps = round(np.median(np.array(fpeses)))
        #     median_fps = float(np.median(np.array(fpeses)))
        #     fps = round(median_fps, 2)
        #     print('fps set: ', fps)
        #     fpeses.append(fps)

        if not ret:
            break

    # Cleanup when closed
    cv2.destroyAllWindows()
    cap.release()
