#!/usr/bin/env python

import sys
import cv2
import numpy as np
import time
import signal
import skimage.measure as measure
import logging
import argparse


TRESHOLD = 5
BLUR_RADIUS = 5


gaussian_ksize = ((BLUR_RADIUS|1),) * 2


def autocrop(image, labels, label):
    rows = np.where(np.any(labels == label, 0))[0]
    if rows.size:
        cols = np.where(np.any(labels == label, 1))[0]
        image = image[cols[0]: cols[-1] + 1, rows[0]: rows[-1] + 1]
    else:
        image = image[:1, :1]

    return image


def check_positive(value):
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError(
            "%s is an invalid positive int value" % value)
    return ivalue


def check_nonnegative(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(
            "%s is an invalid non-negative int value" % value)
    return ivalue


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
    prev_time = time.time()

    while True:
        rval, frame = vc.read()
        if not rval:
            break

        # Frame preprocessing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, gaussian_ksize, 0)
        ret, bw = cv2.threshold(blurred, TRESHOLD, 255, cv2.THRESH_BINARY)
        assert ret

        # Frame feature detection
        labels, count = measure.label(bw, return_num=True)
        ctr += count
        new_time = time.time()
        logger.debug("count=%d fps=%.2f res=%s", ctr, (1/(new_time - prev_time)), frame.shape[:-1])
        if count:
            filename = "out_%.2d.bmp" % (out,)
            cv2.imwrite(filename, frame)
            logger.info("wrote file %s with %d particle(s)", repr(filename), count)
            for label in np.unique(labels):
                if label == 0:
                    continue
                #label_mask = np.zeros_like(frame)
                #label_mask[labels == label] = (255,) * frame.shape[2]
                #masked_frame = cv2.bitwise_and(frame, label_mask)
                cropped_frame = autocrop(frame, labels, label)
                cv2.imwrite("out_%.2d_%.2d.bmp" % (out, label), cropped_frame)
            out += 1
        prev_time = new_time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-W",
                        "--width",
                        help="camera resolution: width",
                        type=check_positive,
                        default=1920)
    parser.add_argument("-H",
                        "--height",
                        help="camera resolution: height",
                        type=check_positive,
                        default=960)
    parser.add_argument("-r",
                        "--fps",
                        help="camera frames per second",
                        type=check_positive,
                        default=30)
    parser.add_argument("-c",
                        "--camera",
                        help="video input index",
                        type=check_nonnegative,
                        default=0)
    parser.add_argument("-d",
                        "--debug",
                        action='store_true',
                        help="debug output")
    return parser.parse_args()


def main():
    signal.signal(signal.SIGINT, exit_handler)
    args = parse_args()
    logger = setup_logger(logging.DEBUG if args.debug else logging.INFO)

    vc = cv2.VideoCapture(args.camera)
    vc.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, args.width)
    vc.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, args.height)
    vc.set(cv2.cv.CV_CAP_PROP_FPS, args.fps)

    if not vc.isOpened():
        raise Exception("Unable to open camera")

    processing_loop(vc, logger)


if __name__ == '__main__':
    main()
