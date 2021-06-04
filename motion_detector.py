
import cv2
import numpy as np
import threading
import datetime
import os
import sqlite3


class Detector:
    def __init__(self, name, password, ip, width, height, threshold, number, channel):
        self.name = name
        self.password = password
        self.ip = ip
        self.width = width
        self.height = height
        self.threshold = threshold
        self.number = number
        self.channel = channel
        print(channel)
    def create_camera(self):  # # rtsp://admin:qwerty123@192.168.88.100:554/ch01/0
        rtsp = "rtsp://" + self.name + ":" + self.password + "@" + self.ip + ":554/" + self.channel + "/1"
        cap = cv2.VideoCapture(rtsp)
        self.width = int(cap.get(3))  # ID number for width is 3
        self.height = int(cap.get(4))  # ID number for height is 480
        cap.set(10, 100)  # ID number for brightness is 10
        return cap
    def detector(self):
        # conn = sqlite3.connect("mydatabase.db")
        # cursor = conn.cursor()
        cam1 = self.create_camera()
        Main_screen = np.zeros(((self.height * 2), (self.width * 2), 3),
                               np.uint8)  # create screen on which all four camera will be stiched
        def read_camera():
            success, current_screen = cam1.read()
            Main_screen[:self.height, :self.width, :3] = current_screen
            return Main_screen
        kernal = np.ones((5, 5), np.uint8)  # form a 5x5 matrix with all ones range is 8-bit
        switch = 0
        timer = 350
        dvizh = 0
        result = 0
        clock = datetime.datetime.now()
        today = clock.strftime("%Y-%m-%d")
        dlist = os.listdir('all')
        yes = 0
        for i in dlist:
            if i == today:
                yes = 1
        if not yes:
            os.mkdir(os.path.join('all', today))
            temp = 'all' + '\\' + today
            os.mkdir(os.path.join(temp, 'ch01'))
            os.mkdir(os.path.join(temp, 'ch04'))
            os.mkdir(os.path.join(temp, 'ch06'))
            os.mkdir(os.path.join(temp, 'ch07'))
            os.mkdir(os.path.join(temp, 'ch08'))
        while True:
            frame1 = read_camera()  # Read the first frame
            grayImage_F1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)  # Convert to gray
            frame2 = read_camera()  # Read the 2nd frame
            grayImage_F2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            diffImage = cv2.absdiff(grayImage_F1, grayImage_F2)  # get the differance --this is cool
            blurImage = cv2.GaussianBlur(diffImage, (5, 5), 0)
            _, thresholdImage = cv2.threshold(blurImage, 20, 255, cv2.THRESH_BINARY)
            dilatedImage = cv2.dilate(thresholdImage, kernal, iterations=5)
            contours, _ = cv2.findContours(dilatedImage, cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)  # find contour is a magic function
            for contour in contours:  # for every change that is detected
                now = datetime.datetime.now()
                name = now.strftime("%d-%m-%Y-%H-%M-%S")
                _time = now.strftime("%H-%M-%S")
                (x, y, w, h) = cv2.boundingRect(contour)  # get the location where change was found
                if cv2.contourArea(contour) > self.threshold:
                    # cv2.rectangle(frame1, (x, y), (x + w, y + h), (255, 0, 0), 1)
                    dvizh = 1
                    if switch == 0:
                        temp = ""
                        if self.channel == "ch01":
                            temp = 'all' + '/' + today + "/" + "ch01"
                        elif self.channel == "ch04":
                            temp = 'all' + '/' + today + "/" + "ch04"
                        elif self.channel == "ch06":
                            temp = 'all' + '/' + today + "/" + "ch06"
                        elif self.channel == "ch07":
                            temp = 'all' + '/' + today + "/" + "ch07"
                        elif self.channel == "ch08":
                            temp = 'all' + '/' + today + "/" + "ch08"
          result = cv2.VideoWriter(os.path.join(temp, str(self.channel) + "-" + name + '.avi'),
                                                 cv2.VideoWriter_fourcc(*'MJPG'),
                                                 10.0, (self.width, self.height))
                        name = now.strftime("%d-%m-%Y-%H-%M-%S")
                        _time = now.strftime("%Y-%m-%d %H:%M:%S")
                        switch = 1
                else:
                    dvizh = 0
            display_screen = frame1[0:self.height, 0:self.width]
            print(dvizh)
            if switch == 1:
                result.write(display_screen)
                if 285 < timer < 300 and dvizh:
                    timer += 300
                if timer > 0:
                    timer -= 1
                else:
                    switch = 0
                    timer = 300
                    result.release()
            if cv2.waitKey(1) & 0xFF == ord('p'):
                cam1.release()
                cv2.destroyAllWindows()
                break

# rtsp://admin:qwerty123@192.168.88.100:554/ch01/0
if __name__ == "__main__":
    while True:
        try:
            channel1 = Detector('admin', "qwerty123", "192.168.88.100", 640, 480, 3000, 1, "ch01")
            channel4 = Detector('admin', "qwerty123", "192.168.88.100", 640, 480, 5000, 1, "ch04")
            channel6 = Detector('admin', "qwerty123", "192.168.88.100", 640, 480, 3000, 1, "ch06")
            channel7 = Detector('admin', "qwerty123", "192.168.88.100", 640, 480, 5000, 1, "ch07")
            channel8 = Detector('admin', "qwerty123", "192.168.88.100", 640, 480, 3000, 1, "ch08")
            ch1 = threading.Thread(target=channel1.detector, daemon=True)
            ch4 = threading.Thread(target=channel4.detector, daemon=True)
            ch6 = threading.Thread(target=channel6.detector, daemon=True)
            ch7 = threading.Thread(target=channel7.detector, daemon=True)
            ch8 = threading.Thread(target=channel8.detector, daemon=True)
            ch1.start()
            ch4.start()
            ch6.start()
            ch7.start()
            ch8.start()
            if not ch1.is_alive():
                print("llllllllll")
                ch1.start()
                ch1.join()
            if not ch4.is_alive():
                print("llllllllll")
                ch4.start()
                ch4.join()
            if not ch6.is_alive():
                print("llllllllll")
                ch6.start()
                ch6.join()
            if not ch7.is_alive():
                print("llllllllll")
                ch7.start()
                ch7.join()
            if not ch8.is_alive():
                print("llllllllll")
                ch8.start()
                ch8.join()
            threading.enumerate()
            ch1.join()
            ch4.join()
            ch6.join()
            ch7.join()
            ch8.join()
        except TypeError:
            break