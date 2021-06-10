import cv2
import os
import numpy as np
import json

from rectangles import select_object


def read_door_info(name=None):
    door_info = {}
    with open(name, 'r') as file:
        lines = file.readlines()
    for line in lines:
        line_l = line.split(";")
        val = line_l[1][1:-2].split(",")
        for i, v in enumerate(val):
            val[i] = int(v)
        door_info[line_l[0]] = val
    return door_info


if __name__ == "__main__":
    folder = 'data_files/videos_motion/test_videos'
    files = os.listdir(folder)
    doors_dict = {}
    around_doors_dict = {}
    action_name1 = 'doors contour'
    action_name2 = 'around doors (where to start writing)'

    for file in files:
        print('', action_name1)
        file_path = os.path.join(folder, file)
        video_capture = cv2.VideoCapture(file_path)
        ret, frame = video_capture.read()
        first_frame = frame.copy()
        cv2.putText(first_frame, "{}".format(action_name1), (10, 35), 0,
                    2e-3 * first_frame.shape[0], (0, 0, 255), 3)

        door = select_object(first_frame)
        if len(door) != 0:
            door = door[0]
        doors_dict[file] = door

        print('', action_name2)
        cv2.putText(frame, "{}".format(action_name2), (10, 35), 0,
                    2e-3 * frame.shape[0], (0, 0, 255), 3)
        door_around = select_object(frame)
        if len(door_around) != 0:
            door_around = door_around[0]
        around_doors_dict[file] = door_around



    mode = 'a' if os.path.exists('cfg/doors_info.json') else 'w'
    with open('cfg/doors_info.json', 'w') as f:
        json.dump(doors_dict, f)

    mode1 = 'a' if os.path.exists('cfg/around_doors_info.json') else 'w'
    with open('cfg/around_doors_info.json', 'w') as wr:
        json.dump(around_doors_dict, wr)

    print(doors_dict)
    print(around_doors_dict)
