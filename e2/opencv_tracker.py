import cv2
import sys

if __name__ == "__main__":
    tracker = cv2.TrackerCSRT_create()
    video = cv2.VideoCapture('1.mp4')
    ret, frame = video.read()
    if not ret:
        sys.exit(0)
    BoundingBox = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
    init_status = tracker.init(frame, BoundingBox)
    while True:
        ret, frame = video.read()
        if not ret:
            break
        status, bbox = tracker.update(frame)
        if status:
            start_p = (int(bbox[0]), int(bbox[1]))
            end_p = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            track_result = cv2.rectangle(frame, start_p, end_p, (0, 0, 255))
            cv2.imshow('Frame', track_result)
            cv2.waitKey(1)
    video.release()
