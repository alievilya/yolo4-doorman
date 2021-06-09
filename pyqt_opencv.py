import os
import cv2
import gc
from multiprocessing import Process, Manager


# Write data to the shared buffer stack:
def write(stack, cam, top: int) -> None:
    """
         :param cam: camera parameters
         :param stack: Manager.list object
         :param top: buffer stack capacity
    :return: None
    """
    print('Process to write: %s' % os.getpid())
    cap = cv2.VideoCapture(cam)
    while True:
        _, img = cap.read()
        if _:
            stack.append(img)
            # Clear the buffer stack every time it reaches a certain capacity
            # Use the gc library to manually clean up memory garbage to prevent memory overflow
            if len(stack) >= top:
                del stack[:]
                gc.collect()


# Read data in the buffer stack:
def read(stack) -> None:
    print('Process to read: %s' % os.getpid())
    while True:
        if len(stack) != 0:
            value = stack.pop()
            cv2.imshow("img", value)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break


if __name__ == '__main__':
    # The parent process creates a buffer stack and passes it to each child process:
    q = Manager().list()
    q2 = Manager().list()
    pw = Process(target=write, args=(q, "rtsp://admin:admin@192.168.1.18:554/1/h264major", 100))
    pr = Process(target=read, args=(q,))

    pw2 = Process(target=write, args=(q2, "rtsp://admin:admin@192.168.1.18:554/2/h264major", 100))
    pr2 = Process(target=read, args=(q2,))
    # Start the child process pw, write:
    pw.start()
    pw2.start()
    # Start the child process pr, read:
    pr.start()
    pr2.start()

    # Wait for pr to end:
    pr.join()

    # pw Process is an infinite loop, can not wait for its end, can only be forced to terminate:
    pw.terminate()
