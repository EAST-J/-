##使用Meanshift算法完成追踪
import cv2
import numpy as np


def init_window(frame):
    BoundingBox = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
    x, y, w, h = BoundingBox
    roi = frame[y:y + h, x:x + w]
    return roi, BoundingBox


def show_img(img):
    cv2.imshow('test', img)
    cv2.waitKey(1)
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    video = cv2.VideoCapture('1.mp4')
    ret, frame = video.read()  ###load video
    if ret:
        print(frame.shape)
        roi,track_window = init_window(frame)
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
        roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
        term = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)  ###设置停止条件
    while (True):
        ret, frame = video.read()  ###load video
        if not ret:
            break
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv_frame], [0], roi_hist, [0, 180], 1)
        ret, track_window = cv2.meanShift(dst, track_window, term)
        x, y, w, h = track_window
        track_result = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255))
        show_img(track_result)
    video.release()
