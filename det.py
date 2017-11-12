#!/usr/bin/env python

import sys
import cv2
import numpy as np
import time
import signal
import skimage.measure as measure


TRESHOLD = 5


def exit_handler(signum, frame):
    sys.exit(0)


def main():
    signal.signal(signal.SIGINT, exit_handler)

    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)
    vc.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 1920)
    vc.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 960)
    vc.set(cv2.cv.CV_CAP_PROP_FPS, 30)

    ctr = 0

    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
        prev_time = time.time()
    else:
        rval = False

    while rval:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, bw = cv2.threshold(gray, TRESHOLD, 255, cv2.THRESH_BINARY)
        assert ret

        labels, count = measure.label(bw, return_num=True)
        ctr += count
        ##ctr += (gray > 5).sum()
        #ctr += np.count_nonzero(frame[:, :] > TRESHOLD)
        new_time = time.time()
        print ctr, "fps=%.2f" % (1/(new_time - prev_time))
        prev_time = new_time
        rval, frame = vc.read()


if __name__ == '__main__':
    main()
