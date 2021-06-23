import json
import os

import cv2

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
    # with open('cfg/camera_link_to_id.json') as json_file:
    #     link2id_dict = json.load(json_file)

    folder = 'data_files/videos_motion/videos_settings'
    files = os.listdir(folder)
    files = [f for f in files if f.endswith('.mp4')]
    doors_dict = {}
    around_doors_dict = {}
    action_name1 = 'doors contour'
    action_name2 = 'around doors (where to start writing)'

    for file in files:
        print('', action_name1)
        meta_name = "data_files/videos_meta/videos_settings/" + file[:-4] + ".json"
        with open(meta_name) as meta_config_json:
            meta_config = json.load(meta_config_json)

        file_path = os.path.join(folder, file)
        cam_id = meta_config["camera_id"]
        video_capture = cv2.VideoCapture(file_path)
        ret, frame = video_capture.read()
        first_frame = frame.copy()
        cv2.putText(first_frame, "{}".format(action_name1), (10, 35), 0,
                    2e-3 * first_frame.shape[0], (0, 0, 255), 3)
        xx = doors_dict.get(cam_id)
        if xx is not None:
            continue
        door = select_object(first_frame)
        if len(door) != 0:
            door = door[0]
        doors_dict[cam_id] = door

        print('', action_name2)
        cv2.putText(frame, "{}".format(action_name2), (10, 35), 0,
                    2e-3 * frame.shape[0], (0, 0, 255), 3)
        door_around = select_object(frame)
        if len(door_around) != 0:
            door_around = door_around[0]
        around_doors_dict[cam_id] = door_around

    mode = 'a' if os.path.exists('cfg/doors_info.json') else 'w'
    with open('cfg/doors_info.json', 'w') as f:
        json.dump(doors_dict, f)

    mode1 = 'a' if os.path.exists('cfg/around_doors_info.json') else 'w'
    with open('cfg/around_doors_info.json', 'w') as wr:
        json.dump(around_doors_dict, wr)

    print(doors_dict)
    print(around_doors_dict)
