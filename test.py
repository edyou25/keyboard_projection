import os

font_dir = "/usr/share/fonts/truetype/dejavu"
if os.path.isdir(font_dir):
    os.environ["QT_QPA_FONTDIR"] = font_dir

import cv2
import json
import numpy as np

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

IMAGE_SAVE_PATH = "input/test.png"
JSON_SAVE_PATH = "output/out.json"


def compute_bev_transform(corners_xy):
    src = np.array(corners_xy, dtype=np.float32)

    tl, tr, br, bl = src

    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    bev_w = int(max(width_top, width_bottom))

    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    bev_h = int(max(height_left, height_right))

    dst = np.array(
        [
            [0, 0],
            [bev_w - 1, 0],
            [bev_w - 1, bev_h - 1],
            [0, bev_h - 1],
        ],
        dtype=np.float32,
    )

    M = cv2.getPerspectiveTransform(src, dst)
    return M, bev_w, bev_h


def save_transform_json(corners_xy, M, bev_w, bev_h):
    os.makedirs(os.path.dirname(JSON_SAVE_PATH), exist_ok=True)

    data = {
        "corners_order": [
            "top_left",
            "top_right",
            "bottom_right",
            "bottom_left",
        ],
        "corners_xy": np.array(corners_xy).tolist(),
        "perspective_transform_matrix": M.tolist(),
        "bev_size": {
            "width": int(bev_w),
            "height": int(bev_h),
        },
    }

    with open(JSON_SAVE_PATH, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved transform json to: {JSON_SAVE_PATH}")


def manual_select_corners(image):
    points = []

    max_display_width = 1280
    h, w = image.shape[:2]
    scale = min(1.0, max_display_width / w)

    display_w = int(w * scale)
    display_h = int(h * scale)

    def redraw():
        vis = cv2.resize(image, (display_w, display_h))

        for i, pt in enumerate(points):
            x = int(pt[0] * scale)
            y = int(pt[1] * scale)

            cv2.circle(vis, (x, y), 6, (0, 0, 255), -1)
            cv2.putText(
                vis,
                str(i + 1),
                (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

        if len(points) >= 2:
            pts = np.array(points, dtype=np.float32) * scale
            pts = pts.astype(np.int32)
            for i in range(len(pts) - 1):
                cv2.line(vis, tuple(pts[i]), tuple(pts[i + 1]), (0, 255, 0), 2)

            if len(pts) == 4:
                cv2.line(vis, tuple(pts[3]), tuple(pts[0]), (0, 255, 0), 2)

        cv2.putText(
            vis,
            "Click: TL -> TR -> BR -> BL | r: reset | Enter: confirm | q: quit",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        return vis

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            original_x = x / scale
            original_y = y / scale
            points.append([original_x, original_y])
            print(f"Point {len(points)}: ({original_x:.1f}, {original_y:.1f})")

    win_name = "Select 4 corners"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
    cv2.setMouseCallback(win_name, mouse_callback)

    while True:
        cv2.imshow(win_name, redraw())
        key = cv2.waitKey(20) & 0xFF

        if key == ord("r"):
            points.clear()
            print("Reset points")

        elif key == ord("q"):
            cv2.destroyWindow(win_name)
            return None

        elif key == 13 or key == 10:
            if len(points) == 4:
                cv2.destroyWindow(win_name)
                return points
            else:
                print("Need exactly 4 points")

    return None


def main():
    cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)
    cap.set(cv2.CAP_PROP_FPS, 90)

    if not cap.isOpened():
        print("Cannot open camera")
        return

    frame = None
    for _ in range(10):
        ret, frame = cap.read()

    if not ret or frame is None:
        print("Cannot read frame")
        cap.release()
        return

    # os.makedirs(os.path.dirname(IMAGE_SAVE_PATH), exist_ok=True)
    # cv2.imwrite(IMAGE_SAVE_PATH, frame)
    # print(f"Saved image to: {IMAGE_SAVE_PATH}")
    # print("Frame shape:", frame.shape)

    corners_xy = manual_select_corners(frame)
    if corners_xy is None:
        print("Corner selection cancelled")
        cap.release()
        cv2.destroyAllWindows()
        return

    M, bev_w, bev_h = compute_bev_transform(corners_xy)
    save_transform_json(corners_xy, M, bev_w, bev_h)

    cv2.namedWindow("BEV", cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        bev = cv2.warpPerspective(frame, M, (bev_w, bev_h))

        # 如果方向反了，可以保留这一行；如果方向正常，就注释掉
        bev = cv2.rotate(bev, cv2.ROTATE_180)

        cv2.imshow("BEV", bev)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()