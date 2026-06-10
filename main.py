#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Detect keyboard 4 corners and automatically generate a BEV image.

Pipeline:
1. YOLO-World: detect keyboard bbox with text prompts
2. SAM2: segment keyboard mask from the bbox
3. OpenCV: estimate 4 projected corners of the keyboard
4. Perspective transform: warp the keyboard to a rectangle (BEV)

Run:
    python detect_keyboard_bev.py
"""

import json
from pathlib import Path
import os
import cv2
import numpy as np
from ultralytics import YOLO, SAM



# 输入图像
IMAGE_PATH = "input/test.png"

# 输出文件
VIS_OUT_PATH = "output/detection.png"      # 检测可视化图
BEV_OUT_PATH = "output/bev.png"         # 透视矫正后的俯视图
JSON_OUT_PATH = "output/out.json"    # 角点、矩阵等信息

# YOLO-World
YOLO_WORLD_MODEL = "models/yolov8s-worldv2.pt"
DET_CONF = 0.05
IMG_SIZE = 960

# SAM2
SAM_MODEL = "models/sam2.1_b.pt"

# 四角点拟合策略
# True: 优先用轮廓四边形
# False: 直接用 minAreaRect（遮挡场景更稳）
USE_APPROX_POLYGON = False

# 是否对 mask 做闭运算
USE_MORPH_CLOSE = True


# ============================================================
# 工具函数
# ============================================================

def order_points_clockwise(points: np.ndarray) -> np.ndarray:
    """
    将四个点排序为：
    top-left, top-right, bottom-right, bottom-left
    """
    points = np.asarray(points, dtype=np.float32)

    s = points.sum(axis=1)
    diff = np.diff(points, axis=1).reshape(-1)

    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = points[np.argmin(s)]       # top-left
    ordered[2] = points[np.argmax(s)]       # bottom-right
    ordered[1] = points[np.argmin(diff)]    # top-right
    ordered[3] = points[np.argmax(diff)]    # bottom-left

    return ordered


def detect_keyboard_bbox(image_path: str):
    """
    用 YOLO-World 检测键盘 bbox
    """
    model = YOLO(YOLO_WORLD_MODEL)

    model.set_classes([
        "keyboard",
        "computer keyboard",
        "mechanical keyboard",
        "laptop keyboard",
    ])

    results = model.predict(
        image_path,
        conf=DET_CONF,
        imgsz=IMG_SIZE,
        verbose=False,
    )

    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        raise RuntimeError(
            "No keyboard detected. "
            "Try lowering DET_CONF or using a clearer image."
        )

    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()

    best_idx = int(np.argmax(scores))
    best_box = boxes[best_idx]
    best_score = float(scores[best_idx])

    return best_box, best_score


def segment_keyboard_mask(image_path: str, bbox_xyxy: np.ndarray):
    """
    用 SAM2 根据 bbox 分割键盘 mask
    """
    sam = SAM(SAM_MODEL)

    bbox = bbox_xyxy.astype(float).tolist()

    results = sam(
        image_path,
        bboxes=bbox,
        verbose=False,
    )

    result = results[0]

    if result.masks is None or result.masks.data is None or len(result.masks.data) == 0:
        raise RuntimeError("SAM2 failed to produce mask.")

    masks = result.masks.data.cpu().numpy()

    # 若有多个 mask，选面积最大的
    areas = masks.reshape(masks.shape[0], -1).sum(axis=1)
    best_idx = int(np.argmax(areas))
    mask = masks[best_idx]

    return mask


def mask_to_corners(mask: np.ndarray, image_shape):
    """
    从 mask 中估计键盘四角点
    返回:
        corners: [TL, TR, BR, BL]
        binary:  二值 mask
    """
    h, w = image_shape[:2]

    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    binary = (mask > 0.5).astype(np.uint8) * 255

    if USE_MORPH_CLOSE:
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if len(contours) == 0:
        raise RuntimeError("No contour found from mask.")

    contour = max(contours, key=cv2.contourArea)

    area = cv2.contourArea(contour)
    if area < 100:
        raise RuntimeError(
            f"Keyboard contour too small: area={area:.1f}. "
            "Detection or segmentation may have failed."
        )

    # 方法1：优先四边形拟合
    if USE_APPROX_POLYGON:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        if len(approx) == 4:
            corners = approx.reshape(4, 2).astype(np.float32)
            corners = order_points_clockwise(corners)
            return corners, binary

    # 方法2：最小外接旋转矩形（遮挡更稳）
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    corners = order_points_clockwise(box.astype(np.float32))

    return corners, binary


def warp_to_bev(image: np.ndarray, corners: np.ndarray):
    """
    使用四角点把图像中的键盘四边形矫正成矩形 BEV

    corners 顺序必须是:
        top-left, top-right, bottom-right, bottom-left

    返回:
        bev_image
        M
        target_width
        target_height
    """
    corners = corners.astype(np.float32)
    tl, tr, br, bl = corners

    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    target_width = int(round(max(width_top, width_bottom)))

    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    target_height = int(round(max(height_left, height_right)))

    target_width = max(target_width, 1)
    target_height = max(target_height, 1)

    dst = np.array(
        [
            [0, 0],
            [target_width - 1, 0],
            [target_width - 1, target_height - 1],
            [0, target_height - 1],
        ],
        dtype=np.float32,
    )

    M = cv2.getPerspectiveTransform(corners, dst)
    bev_image = cv2.warpPerspective(image, M, (target_width, target_height))
    bev_image = cv2.rotate(bev_image, cv2.ROTATE_180)
    return bev_image, M, target_width, target_height


def draw_visualization(
    image: np.ndarray,
    bbox_xyxy: np.ndarray,
    corners: np.ndarray,
    mask_binary: np.ndarray,
    det_score: float,
    out_path: str,
):
    """
    绘制检测结果：
    - bbox
    - mask
    - 四角点
    """
    vis = image.copy()

    # 半透明 mask
    overlay = vis.copy()
    overlay[mask_binary > 0] = (0, 255, 0)
    vis = cv2.addWeighted(overlay, 0.25, vis, 0.75, 0)

    # bbox
    x1, y1, x2, y2 = bbox_xyxy.astype(int)
    cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # 四边形
    pts = corners.astype(int)
    cv2.polylines(vis, [pts], isClosed=True, color=(0, 0, 255), thickness=3)

    names = ["TL", "TR", "BR", "BL"]
    for name, p in zip(names, pts):
        x, y = int(p[0]), int(p[1])
        cv2.circle(vis, (x, y), 6, (0, 0, 255), -1)
        cv2.putText(
            vis,
            name,
            (x + 6, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        vis,
        f"keyboard score={det_score:.3f}",
        (x1, max(25, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 0, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.imwrite(out_path, vis)


def save_result_json(
    image_path: str,
    bbox_xyxy: np.ndarray,
    det_score: float,
    corners: np.ndarray,
    M: np.ndarray,
    bev_width: int,
    bev_height: int,
    json_path: str,
):
    """
    保存结果到 json
    """
    data = {
        "image": image_path,
        "bbox_xyxy": bbox_xyxy.astype(float).tolist(),
        "det_score": float(det_score),
        "corners_order": [
            "top_left",
            "top_right",
            "bottom_right",
            "bottom_left",
        ],
        "corners_xy": corners.astype(float).tolist(),
        "perspective_transform_matrix": M.astype(float).tolist(),
        "bev_size": {
            "width": int(bev_width),
            "height": int(bev_height),
        },
    }
    dir_name = os.path.dirname(json_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return data


import time


def main():
    total_t0 = time.perf_counter()

    t0 = time.perf_counter()
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {IMAGE_PATH}")
    print(f"[0/5] Read image done, time: {time.perf_counter() - t0:.3f}s")

    t0 = time.perf_counter()
    print("[1/5] Detecting keyboard bbox...")
    bbox, score = detect_keyboard_bbox(IMAGE_PATH)
    print(f"[1/5] Detecting keyboard bbox done, time: {time.perf_counter() - t0:.3f}s")

    t0 = time.perf_counter()
    print("[2/5] Segmenting keyboard mask...")
    mask = segment_keyboard_mask(IMAGE_PATH, bbox)
    print(f"[2/5] Segmenting keyboard mask done, time: {time.perf_counter() - t0:.3f}s")

    t0 = time.perf_counter()
    print("[3/5] Estimating 4 keyboard corners...")
    corners, mask_binary = mask_to_corners(mask, image.shape)
    print(f"[3/5] Estimating 4 keyboard corners done, time: {time.perf_counter() - t0:.3f}s")

    t0 = time.perf_counter()
    print("[4/5] Saving visualization...")
    draw_visualization(
        image=image,
        bbox_xyxy=bbox,
        corners=corners,
        mask_binary=mask_binary,
        det_score=score,
        out_path=VIS_OUT_PATH,
    )
    print(f"[4/5] Saving visualization done, time: {time.perf_counter() - t0:.3f}s")

    t0 = time.perf_counter()
    print("[5/5] Generating BEV image...")
    bev_image, M, bev_w, bev_h = warp_to_bev(image, corners)
    cv2.imwrite(BEV_OUT_PATH, bev_image)
    print(f"[5/5] Generating BEV image done, time: {time.perf_counter() - t0:.3f}s")

    print(f"[Total] Pipeline time: {time.perf_counter() - total_t0:.3f}s")

    data = save_result_json(
        image_path=IMAGE_PATH,
        bbox_xyxy=bbox,
        det_score=score,
        corners=corners,
        M=M,
        bev_width=bev_w,
        bev_height=bev_h,
        json_path=JSON_OUT_PATH,
    )

    print("\nDone.")
    print(f"Visualization : {VIS_OUT_PATH}")
    print(f"BEV image     : {BEV_OUT_PATH}")
    print(f"Result json   : {JSON_OUT_PATH}")

    print("\nCorners:")
    for name, point in zip(data["corners_order"], data["corners_xy"]):
        print(f"  {name}: ({point[0]:.1f}, {point[1]:.1f})")

    print("\nPerspective Transform Matrix:")
    print(np.array(data["perspective_transform_matrix"]))


if __name__ == "__main__":
    main()