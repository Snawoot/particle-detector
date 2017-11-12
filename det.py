#!/usr/bin/env python

import sys
import cv2
import numpy as np
import time
import signal
import skimage.measure as measure
import logging


TRESHOLD = 5
BLUR_RADIUS = 4


gaussian_ksize = (BLUR_RADIUS,) * 4


def setup_logger(level=logging.INFO):
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    log_handler = logging.StreamHandler(sys.stderr)
    log_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(log_handler)
    return logger


def exit_handler(signum, frame):
    sys.exit(0)


def processing_loop(vc, logger):
    ctr = 0
    out = 0

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
        new_time = time.time()
        logger.debug("count=%d fps=%.2f res=%s", ctr, (1/(new_time - prev_time)), frame.shape[:-1])
        if count:
            cv2.imwrite("out_%.2d.bmp" % (out,), frame)
            out += 1
        prev_time = new_time
        rval, frame = vc.read()


def main():
    signal.signal(signal.SIGINT, exit_handler)
    logger = setup_logger(logging.DEBUG)

    vc = cv2.VideoCapture(0)
    vc.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 1920)
    vc.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 960)
    vc.set(cv2.cv.CV_CAP_PROP_FPS, 30)

    processing_loop(vc, logger)


if __name__ == '__main__':
    main()
