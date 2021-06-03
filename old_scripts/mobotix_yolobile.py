import sys
from time import gmtime
import imutils.video
import telebot
from deep_sort_pytorch.deep_sort import DeepSort
from deep_sort_pytorch.utils.parser import get_config
from models import *  # set ONNX_EXPORT in models.py
from tracking_modules import Counter, Writer
from tracking_modules import find_centroid, Rectangle, bbox_rel, draw_boxes, select_object, \
    find_ratio_ofbboxes
from utils.datasets import *
from utils.utils import *
import sys

from collections import OrderedDict
import imutils.video

sys.path.append('/venv/lib/python3.7/site-packages/')

#pip3 install torch==1.7.0 torchvision==0.8.1 -f https://download.pytorch.org/whl/cu101/torch_stable.html



def detect(config):
    sent_videos = set()
    fpeses = []
    fps = 0


from tracking_modules import find_centroid, Rectangle, bbox_rel, draw_boxes, find_ratio_ofbboxes
from utils.datasets import *
from utils.utils import *

sys.path.append('/venv/lib/python3.7/site-packages/')

# pip3 install torch==1.7.0 torchvision==0.8.1 -f https://download.pytorch.org/whl/cu101/torch_stable.html

def detect(config):
    # sent_videos = set()
    TIME_TO_SEND_MSG = 10  # Greenvich Time
    months_rus = ('января', 'февраля', 'марта', 'апреля',
                  'мая', 'июня', 'июля','августа',
                  'сентября', 'октября','ноября', 'декабря')
    fpeses = []
    fps = 0.0
    fps_imutils = imutils.video.FPS().start()

    left_array = None
    rect_left = None

    token = "xxx"
    bot = telebot.TeleBot(token)

    def send_message(current_date, counter_in, counter_out):
        channel = '-1001399933919'
        msg_tosend = "{}: зашло {}, вышло {}".format(current_date, counter_in, counter_out)
        bot.send_message(chat_id=channel, text=msg_tosend)
    # camera info
    save_img = True
    imgsz = (416, 416) if ONNX_EXPORT else config[
        "img_size"]  # (320, 192) or (416, 256) or (608, 352) for (height, width)
    out, source, weights, half, view_img = config["output"], config["source"], config["weights"], \
                                           config["half"], config["view_img"]
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')
    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(config["config_deepsort"])
    # initial objects of classes
    counter = Counter(counter_in=0, counter_out=0, track_id=0)
    VideoHandler = Writer()

    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize device, weights etc.
    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else config["device"])
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA
    # Initialize model
    model = Darknet(config["cfg"], imgsz)

    # Load weights
    attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'], strict=False)
    else:  # darknet format
        load_darknet_weights(model, weights)
    # Eval mode
    model.to(device).eval()
    # Half precision
    print(half)
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    print(half)
    if half:
        model.half()

    if webcam:
        view_img = True
        torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        view_img = True
        torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = load_classes(config["names"])
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img.float()) if device.type != 'cpu' else None  # run once

    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        if rect_left is None:
            if webcam:  # batch_size >= 1
                im0 = im0s[0].copy()
            else:
                im0 = im0s
            left_array = [0, 0, im0.shape[1] / 2, im0.shape[0]]
            rect_left = Rectangle(left_array[0], left_array[1], left_array[2], left_array[3])

        flag_anyone_in_door = False
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Inference
        t1 = torch_utils.time_synchronized()

        pred = model(img, augment=config["augment"])[0]
        # to float
        if half:
            pred = pred.float()
        # Apply NMS
        classes = None if config["classes"] == "None" else config["classes"]
        pred = non_max_suppression(pred, config["conf_thres"], config["iou_thres"],
                                   multi_label=False, classes=classes, agnostic=config["agnostic_nms"])
        # Process detections
        lost_ids = counter.return_lost_ids()
        for i, det in enumerate(pred):  # detections for image i
            if webcam:  # batch_size >= 1
                im0 = im0s[i].copy()
            else:
                im0 = im0s

            bbox_xywh = []
            confs = []
            if det is not None and len(det):
                # Rescale boxes from imgsz to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # Print results
                for c in det[:, -1].unique():
                    if names[int(c)] not in config["needed_classes"]:
                        continue
                # Write results
                for *xyxy, conf, cls in det:
                    #  check if bbox`s class is needed
                    if names[int(cls)] not in config["needed_classes"]:
                        continue
                    x_c, y_c, bbox_w, bbox_h = bbox_rel(*xyxy)
                    obj = [x_c, y_c, bbox_w, bbox_h]
                    bbox_xywh.append(obj)
                    confs.append([conf.item()])
                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])

        detections = torch.Tensor(bbox_xywh)
        confidences = torch.Tensor(confs)

        if len(detections) != 0:
            outputs_tracked = deepsort.update(detections, confidences, im0)
            counter.someone_inframe()
            # draw boxes for visualization
            if len(outputs_tracked) > 0:
                bbox_xyxy = outputs_tracked[:, :4]
                identities = outputs_tracked[:, -1]
                draw_boxes(im0, bbox_xyxy, identities)
                counter.update_identities(identities)

                for bbox_tracked, id_tracked in zip(bbox_xyxy, identities):

                    ratio_initial = find_ratio_ofbboxes(bbox=bbox_tracked, rect_compare=rect_left)
                    #  чел первый раз в контуре двери
                    if VideoHandler.counter_frames_indoor == 0:
                        VideoHandler.start_video(id_tracked)
                        flag_anyone_in_door = True
                    elif id_tracked not in VideoHandler.id_inside_door_detected:
                        VideoHandler.continue_opened_video(id=id_tracked, seconds=3)
                        flag_anyone_in_door = True

                    if id_tracked not in counter.people_init or counter.people_init[id_tracked] == 0:
                        counter.obj_initialized(id_tracked)
                        if ratio_initial >= 0.8 and bbox_tracked[3] < left_array[3]:
                            counter.people_init[id_tracked] = 2
                        elif ratio_initial < 0.8 and bbox_tracked[3] > left_array[3]:
                            counter.people_init[id_tracked] = 1
                        else:
                            # res is None, means that object is not in door contour
                            counter.people_init[id_tracked] = 1
                        counter.frame_age_counter[id_tracked] = 0

                        counter.people_bbox[id_tracked] = bbox_tracked

                    counter.cur_bbox[id_tracked] = bbox_tracked
        else:
            deepsort.increment_ages()
            if counter.need_to_clear():
                counter.clear_all()

        for val in counter.people_init.keys():
            # check bbox also
            cur_c = find_centroid(counter.cur_bbox[val])
            init_c = find_centroid(counter.people_bbox[val])
            vector_person = (cur_c[0] - init_c[0],
                             cur_c[1] - init_c[1])

            if val in lost_ids and counter.people_init[val] != -1:
                # if vector_person < 0 then current coord is less than initialized, it means that man is going
                # in the exit direction
                ratio = find_ratio_ofbboxes(bbox=counter.cur_bbox[val], rect_compare=rect_left)
                if vector_person[0] > 200 and counter.people_init[val] == 2 \
                        and ratio < 0.7:
                    counter.get_out()
                    VideoHandler.stop_recording(action_occured="вышел из кабинета")
                    print('video {}, action: {}, vector {} \n'.format(VideoHandler.video_name,
                                                                      VideoHandler.action_occured,
                                                                      vector_person))

                elif vector_person[0] < -100 and counter.people_init[val] == 1 \
                        and ratio >= 0.7:
                    counter.get_in()
                    VideoHandler.stop_recording(action_occured="вышел из кабинета")
                    print('video {}, action: {}, vector {} \n'.format(VideoHandler.video_name,
                                                                      VideoHandler.action_occured,
                                                                      vector_person))

                counter.people_init[val] = -1
                lost_ids.remove(val)

            counter.clear_lost_ids()

        ins, outs = counter.show_counter()
        cv2.rectangle(im0, (0, 0), (250, 50),
                      (0, 0, 0), -1, 8)

        cv2.rectangle(im0, (int(left_array[0]), int(left_array[1])), (int(left_array[2]), int(left_array[3])),
                      (23, 158, 21), 3)

        cv2.putText(im0, "in: {}, out: {} ".format(ins, outs), (10, 35), 0,
                    1e-3 * im0.shape[0], (255, 255, 255), 3)

        if VideoHandler.stop_writing(im0):
            # send_new_posts(video_name, action_occured)

            sent_videos.add(VideoHandler.video_name)
            with open('data_files/logs2.txt', 'a', encoding="utf-8-sig") as wr:
                wr.write(
                    'video {}, action: {}, vector {} \n'.format(VideoHandler.video_name, VideoHandler.action_occured,
                                                            vector_person))

            VideoHandler = Writer()
            VideoHandler.set_fps(fps)

        else:
            VideoHandler.continue_writing(im0, flag_anyone_in_door)

        if view_img:
            cv2.imshow('im0', im0)
            if cv2.waitKey(1) == ord('q'):  # q to quit
                raise StopIteration

        delta_time = (torch_utils.time_synchronized() - t1)

        if len(fpeses) < 30:
            fpeses.append(1 / delta_time)
        elif len(fpeses) == 30:
            median_fps = float(np.median(np.array(fpeses)))
            # fps = round(median_fps, 1)
            fps = 20
            print('fps set: ', fps)
            VideoHandler.set_fps(fps)
            counter.set_fps(fps)
            fpeses.append(fps)
            motion_detection = True
        else:
            print('\nflag writing video: ', VideoHandler.flag_writing_video)
            print('flag stop writing: ', VideoHandler.flag_stop_writing)
            print('flag anyone in door: ', flag_anyone_in_door)
            print('counter frames indoor: ', VideoHandler.counter_frames_indoor)
        # fps = 20
        gm_time = gmtime()
        if gm_time.tm_hour == TIME_TO_SEND_MSG and not counter.just_inited:
            day = gm_time.tm_mday
            month = months_rus[gm_time.tm_mon - 1]
            year = gm_time.tm_year
            date = "{} {} {}".format(day, month, year)
            in_counted, out_counted = counter.show_counter()
            send_message(current_date=date, counter_in=in_counted, counter_out=out_counted)
            counter = Counter(0, 0, 0)


# python detect.py --cfg cfg/csdarknet53s-panet-spp.cfg --weights cfg/best14x-49.pt --source 0
import json

if __name__ == '__main__':
    # subprocess.run("python send_video.py", shell=True)
    # os.system("python send_video.py &")
    with open("../cfg/detection_tracker_cfg.json") as detection_config:
        detect_config = json.load(detection_config)
    print(detect_config["cfg"])

    with torch.no_grad():
        detect(config=detect_config)
