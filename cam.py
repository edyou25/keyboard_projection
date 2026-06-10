import cv2
import subprocess


def set_camera_controls():
    commands = [
        ["v4l2-ctl", "-d", "/dev/video0", "-c", "power_line_frequency=1"],   # HK 50Hz
        ["v4l2-ctl", "-d", "/dev/video0", "-c", "auto_exposure=1"],           # manual exposure
        ["v4l2-ctl", "-d", "/dev/video0", "-c", "exposure_time_absolute=200"],
        ["v4l2-ctl", "-d", "/dev/video0", "-c", "gain=0"],
        ["v4l2-ctl", "-d", "/dev/video0", "-c", "brightness=50"],
    ]

    for cmd in commands:
        subprocess.run(cmd, check=False)
import os
font_dir = "/usr/share/fonts/truetype/dejavu"
if os.path.isdir(font_dir):
    os.environ["QT_QPA_FONTDIR"] = font_dir # for somr warnings about missing fonts

cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)
cap.set(cv2.CAP_PROP_FPS, 30)
set_camera_controls()
if not cap.isOpened():
    print("Cannot open camera")
    exit()

cv2.namedWindow("USB Camera", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("USB Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()