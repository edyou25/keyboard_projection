import os

font_dir = "/usr/share/fonts/truetype/dejavu"
if os.path.isdir(font_dir):
    os.environ["QT_QPA_FONTDIR"] = font_dir

import cv2
import json
import numpy as np


JSON_PATH = "output/out.json"


def load_transform_from_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    M = np.array(data["perspective_transform_matrix"], dtype=np.float32)

    bev_w = int(data["bev_size"]["width"])
    bev_h = int(data["bev_size"]["height"])

    return M, bev_w, bev_h


def main():
    M, bev_w, bev_h = load_transform_from_json(JSON_PATH)

    cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("Cannot open camera")
        return
    
    save_path = "input/test.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    frame = None
    for _ in range(10):
        ret, frame = cap.read()

    if ret and frame is not None:
        cv2.imwrite(save_path, frame)
        print(f"Saved image to: {save_path}")
        print("Saved frame shape:", frame.shape)

    # cv2.namedWindow("USB Camera", cv2.WINDOW_NORMAL)
    cv2.namedWindow("BEV", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 根据 JSON 里的矩阵实时计算 BEV
        bev = cv2.warpPerspective(frame, M, (bev_w, bev_h))
        bev = cv2.rotate(bev, cv2.ROTATE_180)

        # cv2.imshow("USB Camera", frame)
        cv2.imshow("BEV", bev)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()