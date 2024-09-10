import os
import cv2
import json
import math
import random
import statistics
import numpy as np
from typing import Dict, List, Tuple
import numpy.typing as npt
from math import ceil
import matplotlib.pyplot as plt
from MonlamOCR.Data import (
    LayoutDetectionConfig,
    Line,
    BBox,
    LineData,
    OCRConfig,
    LineDetectionConfig,
)
from datetime import datetime


def show_image(
    image: npt.NDArray, cmap: str = "", axis="off", fig_x: int = 24, fix_y: int = 13
) -> None:
    plt.figure(figsize=(fig_x, fix_y))
    plt.axis(axis)

    if cmap != "":
        plt.imshow(image, cmap=cmap)
    else:
        plt.imshow(image)


def show_overlay(
    image: npt.NDArray,
    mask: npt.NDArray,
    alpha=0.4,
    axis="off",
    fig_x: int = 24,
    fix_y: int = 13,
):
    plt.figure(figsize=(fig_x, fix_y))
    plt.axis(axis)
    plt.imshow(image)
    plt.imshow(mask, alpha=alpha)


def get_utc_time():
    utc_time = datetime.now()
    utc_time = utc_time.strftime('%Y-%m-%dT%H:%M:%S')

    return utc_time


def get_file_name(file_path: str) -> str:
    name_segments = os.path.basename(file_path).split(".")[:-1]
    name = "".join(f"{x}." for x in name_segments)
    return name.rstrip(".")


def create_dir(dir_name: str) -> None:
    if not os.path.exists(dir_name):
        try:
            os.makedirs(dir_name)
            print(f"Created directory at  {dir_name}")
        except IOError as e:
            print(f"Failed to create directory at: {dir_name}, {e}")


def get_random_color() -> Tuple[int, int, int]:
    rand_r = random.randint(10, 255)
    rand_g = random.randint(10, 200)
    rand_b = random.randint(10, 220)
    color = (rand_r, rand_g, rand_b)

    return color


def resize_to_height(image, target_height: int) -> Tuple[npt.NDArray, float]:
    scale_ratio = target_height / image.shape[0]
    image = cv2.resize(
        image,
        (int(image.shape[1] * scale_ratio), target_height),
        interpolation=cv2.INTER_LINEAR,
    )
    return image, scale_ratio


def resize_to_width(image, target_width: int = 2048) -> Tuple[npt.NDArray, float]:
    scale_ratio = target_width / image.shape[1]
    image = cv2.resize(
        image,
        (target_width, int(image.shape[0] * scale_ratio)),
        interpolation=cv2.INTER_LINEAR,
    )
    return image, scale_ratio


def binarize(
    image: npt.ArrayLike, adaptive: bool = True, block_size: int = 51, c: int = 13
) -> npt.NDArray:
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    if adaptive:
        bw = cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            c,
        )

    else:
        _, bw = cv2.threshold(image, 120, 255, cv2.THRESH_BINARY)

    bw = cv2.cvtColor(bw, cv2.COLOR_GRAY2RGB)
    return bw


def get_paddings(image: npt.NDArray, patch_size: int = 512) -> Tuple[int, int]:
    max_x = ceil(image.shape[1] / patch_size) * patch_size
    max_y = ceil(image.shape[0] / patch_size) * patch_size
    pad_x = max_x - image.shape[1]
    pad_y = max_y - image.shape[0]

    return pad_x, pad_y


def tile_image(padded_img: npt.NDArray, patch_size: int = 512):
    x_steps = int(padded_img.shape[1] / patch_size)
    y_steps = int(padded_img.shape[0] / patch_size)
    y_splits = np.split(padded_img, y_steps, axis=0)

    patches = [np.split(x, x_steps, axis=1) for x in y_splits]
    patches = [x for xs in patches for x in xs]

    return patches, y_steps


def stitch_predictions(prediction: npt.NDArray, y_steps: int) -> npt.NDArray:
    pred_y_split = np.split(prediction, y_steps, axis=0)
    x_slices = [np.hstack(x) for x in pred_y_split]
    concat_img = np.vstack(x_slices)

    return concat_img


def pad_image(
    image: npt.NDArray, pad_x: int, pad_y: int, pad_value: int = 0
) -> npt.NDArray:
    padded_img = np.pad(
        image,
        pad_width=((0, pad_y), (0, pad_x), (0, 0)),
        mode="constant",
        constant_values=pad_value,
    )

    return padded_img


def preprocess_image(
    image: npt.NDArray,
    patch_size: int = 512,
    clamp_width: int = 4096,
    clamp_height: int = 2048,
    clamp_size: bool = True,
):
    """
    Some dimension checking and resizing to avoid very large inputs on which the line(s) on the resulting tiles could be too big and cause troubles with the current line model.
    """
    if clamp_size and image.shape[1] > image.shape[0] and image.shape[1] > clamp_width:
        image, _ = resize_to_width(image, clamp_width)

    elif (
        clamp_size and image.shape[0] > image.shape[1] and image.shape[0] > clamp_height
    ):
        image, _ = resize_to_height(image, clamp_height)

    elif image.shape[0] < patch_size:
        image, _ = resize_to_height(image, patch_size)

    pad_x, pad_y = get_paddings(image, patch_size)
    padded_img = pad_image(image, pad_x, pad_y, pad_value=255)

    return padded_img, pad_x, pad_y


def normalize(image: npt.NDArray) -> npt.NDArray:
    image = image.astype(np.float32)
    image /= 255.0
    return image


def get_contours(
    prediction: npt.NDArray, optimize: bool = True, size_tresh: int = 200
) -> list:
    prediction = np.where(prediction > 200, 255, 0)
    prediction = prediction.astype(np.uint8)

    if np.sum(prediction) > 0:
        contours, _ = cv2.findContours(
            prediction, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        print(f"Contours: {len(contours)}")

        if optimize:
            contours = [optimize_countour(x) for x in contours]
            contours = [x for x in contours if cv2.contourArea(x) > size_tresh]
        return contours
    else:
        print("Returning []")
        return []


def sigmoid(x) -> float:
    return 1 / (1 + np.exp(-x))


def get_rotation_angle_from_lines(
    line_mask: npt.NDArray,
    max_angle: float = 5.0,
    debug_angles: bool = False,
) -> float:
    contours, _ = cv2.findContours(line_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    mask_threshold = (line_mask.shape[0] * line_mask.shape[1]) * 0.001
    contours = [x for x in contours if cv2.contourArea(x) > mask_threshold]
    angles = [cv2.minAreaRect(x)[2] for x in contours]

    low_angles = [x for x in angles if abs(x) != 0.0 and x < max_angle]
    high_angles = [x for x in angles if abs(x) != 90.0 and x > (90 - max_angle)]

    if debug_angles:
        print(f"All Angles: {angles}")

    if len(low_angles) > len(high_angles) and len(low_angles) > 0:
        mean_angle = float(np.mean(low_angles))

    # check for clockwise rotation
    elif len(high_angles) > 0:
        mean_angle = float(-(90 - np.mean(high_angles)))

    else:
        mean_angle = 0.0

    return mean_angle


def get_rotation_angle_from_houghlines(
    image: npt.NDArray, min_length: int = 80
) -> float:
    clahe = cv2.createCLAHE(clipLimit=0.2, tileGridSize=(8, 8))
    cl_img = clahe.apply(image)
    blurred = cv2.GaussianBlur(cl_img, (13, 13), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 19, 11
    )

    lines = cv2.HoughLinesP(
        thresh, 1, np.pi / 180, threshold=130, minLineLength=min_length, maxLineGap=8
    )

    if lines is None or len(lines) == 0:
        print(f"No lines found in image, skipping...")

        return image, 0

    angles = []
    zero_angles = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = math.atan2(y2 - y1, x2 - x1) * 180 / np.pi

        if 5 > abs(angle) > 0:
            angles.append(angle)

        elif int(angle) == 0:
            zero_angles.append(angle)

    if len(angles) != 0:
        avg_angle = statistics.median(angles)
        ratio = len(zero_angles) / len(angles)

        if ratio < 0.5:
            rot_angle = avg_angle
        elif 0.5 < ratio < 0.9:
            rot_angle = avg_angle / 2
        else:
            rot_angle = 0.0
    else:
        print("No angle data found in image.")
        rot_angle = 0

    return rot_angle


def rotate_from_angle(image: npt.NDArray, angle: float) -> npt.NDArray:
    rows, cols = image.shape[:2]
    rot_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)

    rotated_img = cv2.warpAffine(image, rot_matrix, (cols, rows), borderValue=(0, 0, 0))

    return rotated_img


def mask_n_crop(image: npt.NDArray, mask: npt.NDArray) -> npt.NDArray:
    image = image.astype(np.uint8)
    mask = mask.astype(np.uint8)

    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)

    image_masked = cv2.bitwise_and(image, image, mask, mask)
    image_masked = np.delete(
        image_masked, np.where(~image_masked.any(axis=1))[0], axis=0
    )
    image_masked = np.delete(
        image_masked, np.where(~image_masked.any(axis=0))[0], axis=1
    )

    return image_masked


def get_line_threshold(line_prediction: npt.NDArray, slice_width: int = 20):
    """
    This function generates n slices (of n = steps) width the width of slice_width across the bbox of the detected lines.
    The slice with the max. number of contained contours is taken to be the canditate to calculate the bbox center of each contour and
    take the median distance between each bbox center as estimated line cut-off threshold to sort each line segment across the horizontal

    Note: This approach might turn out to be problematic in case of sparsely spread line segments across a page
    """

    x, y, w, h = cv2.boundingRect(line_prediction)
    x_steps = (w // slice_width) // 2

    bbox_numbers = []

    for step in range(1, x_steps + 1):
        x_offset = x_steps * step
        x_start = x + x_offset
        x_end = x_start + slice_width

        _slice = line_prediction[y : y + h, x_start:x_end]
        contours, _ = cv2.findContours(_slice, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        bbox_numbers.append((len(contours), contours))

    sorted_list = sorted(bbox_numbers, key=lambda x: x[0], reverse=True)

    if len(sorted_list) > 0:
        reference_slice = sorted_list[0]

        y_points = []
        n_contours, contours = reference_slice

        if n_contours == 0:
            print("number of contours is 0")
            line_threshold = 0.0
        else:
            for _, contour in enumerate(contours):
                x, y, w, h = cv2.boundingRect(contour)
                y_center = y + (h // 2)
                y_points.append(y_center)

            line_threshold = float(np.median(y_points) // n_contours)
    else:
        line_threshold = 0.0

    return line_threshold


def sort_bbox_centers(bbox_centers: List[Tuple[int, int]], line_threshold: int = 20) -> List:
    sorted_bbox_centers = []
    tmp_line = []

    for i in range(0, len(bbox_centers)):
        if len(tmp_line) > 0:
            for s in range(0, len(tmp_line)):

                # TODO: refactor this to make this calculation an enum to choose between both methods
                # y_diff = abs(tmp_line[s][1] - bbox_centers[i][1])
                """
                I use the mean of the hitherto present line chunks in tmp_line since
                the precalculated fixed threshold can break the sorting if
                there is some slight bending in the line. This part may need some tweaking after
                some further practical review
                """
                ys = [y[1] for y in tmp_line]
                mean_y = np.mean(ys)
                y_diff = abs(mean_y - bbox_centers[i][1])

                if y_diff > line_threshold:
                    tmp_line.sort(key=lambda x: x[0])
                    sorted_bbox_centers.append(tmp_line.copy())
                    tmp_line.clear()

                    tmp_line.append(bbox_centers[i])
                    break
                else:
                    tmp_line.append(bbox_centers[i])
                    break
        else:
            tmp_line.append(bbox_centers[i])

    sorted_bbox_centers.append(tmp_line)

    for y in sorted_bbox_centers:
        y.sort(key=lambda x: x[0])

    sorted_bbox_centers = list(reversed(sorted_bbox_centers))

    return sorted_bbox_centers


def group_line_chunks(sorted_bbox_centers, lines: List[Line]):
    new_line_data = []
    for bbox_centers in sorted_bbox_centers:

        if len(bbox_centers) > 1:  # i.e. more than 1 bbox center in a group
            contour_stack = []

            for box_center in bbox_centers:
                for line_data in lines:
                    if box_center == line_data.center:
                        contour_stack.append(line_data.contour)
                        break

            stacked_contour = np.vstack(contour_stack)
            stacked_contour = cv2.convexHull(stacked_contour)
            # TODO: are both calls necessary?
            x, y, w, h = cv2.boundingRect(stacked_contour)
            _bbox = BBox(x, y, w, h)
            x_center = _bbox.x + (_bbox.w // 2)
            y_center = _bbox.y + (_bbox.h // 2)

            new_line = Line(
                contour=stacked_contour, bbox=_bbox, center=(x_center, y_center)
            )

            new_line_data.append(new_line)

        else:
            for _bcenter in bbox_centers:
                for line_data in lines:
                    if _bcenter == line_data.center:
                        new_line_data.append(line_data)
                        break

    return new_line_data


def sort_lines_by_threshold2(
    line_mask: npt.NDArray,
    lines: List[Line],
    threshold: int = 20,
    calculate_threshold: bool = True,
    group_lines: bool = True,
    debug: bool = False,
):

    bbox_centers = [x.center for x in lines]

    if calculate_threshold:
        line_treshold = get_line_threshold(line_mask)
    else:
        line_treshold = threshold

    if debug:
        print(f"Line threshold: {threshold}")

    sorted_bbox_centers = sort_bbox_centers(bbox_centers, line_threshold=line_treshold)

    if debug:
        print(sorted_bbox_centers)

    if group_lines:
        new_lines = group_line_chunks(sorted_bbox_centers, lines)
    else:
        _bboxes = [x for xs in sorted_bbox_centers for x in xs]

        new_lines = []
        for _bbox in _bboxes:
            for _line in lines:
                if _bbox == _line.center:
                    new_lines.append(_line)

    return new_lines, line_treshold

def build_line_data(contour: npt.NDArray) -> Line:
    x, y, w, h = cv2.boundingRect(contour)
    x_center = x + (w // 2)
    y_center = y + (h // 2)

    bbox = BBox(x, y, w, h)
    return Line(contour, bbox, (x_center, y_center))


def get_text_bbox(lines: List[Line]):
    all_bboxes = [x.bbox for x in lines]
    min_x = min(a.x for a in all_bboxes)
    min_y = min(a.y for a in all_bboxes)

    max_w = max(a.w for a in all_bboxes)
    max_h = all_bboxes[-1].y + all_bboxes[-1].h

    bbox = BBox(min_x, min_y, max_w, max_h)

    return bbox


def extract_line(line: Line, image: npt.NDArray, k_factor: float = 1.2) -> npt.NDArray:
    bbox_h = line.bbox.h

    iterations = 1
    tmp_img = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    cv2.drawContours(tmp_img, [line.contour], -1, (255, 255, 255), -1)
    # TODO: factor in the width so that longer lines get a lower k_size, otherwise the whole thing overshoots
    k_size = int(bbox_h * k_factor)

    morph_rect = cv2.getStructuringElement(
        shape=cv2.MORPH_RECT, ksize=(k_size, int(k_size * 1.5))
    )
    iterations = 1
    tmp_img = cv2.dilate(tmp_img, kernel=morph_rect, iterations=iterations)
    masked_line = mask_n_crop(image, tmp_img)

    return masked_line


def pol2cart(theta, rho):
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y


def cart2pol(x, y):
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return theta, rho


def rotate_contour(cnt, center_x: int, center_y: int, angle: float):
    cnt_norm = cnt - [center_x, center_y]

    coordinates = cnt_norm[:, 0, :]
    xs, ys = coordinates[:, 0], coordinates[:, 1]
    thetas, rhos = cart2pol(xs, ys)

    thetas = np.rad2deg(thetas)
    thetas = (thetas + angle) % 360
    thetas = np.deg2rad(thetas)

    xs, ys = pol2cart(thetas, rhos)

    cnt_norm[:, 0, 0] = xs
    cnt_norm[:, 0, 1] = ys

    cnt_rotated = cnt_norm + [center_x, center_y]
    cnt_rotated = cnt_rotated.astype(np.int32)

    return cnt_rotated


def optimize_countour(cnt, e=0.001):
    epsilon = e * cv2.arcLength(cnt, True)
    return cv2.approxPolyDP(cnt, epsilon, True)


def pad_to_width(
    img: npt.NDArray, target_width: int, target_height: int, padding: str
) -> npt.NDArray:
    
    if len(img.shape) == 3:
        _, _, channels = img.shape
    else:
        channels = 1

    tmp_img, _ = resize_to_width(img, target_width)

    height = tmp_img.shape[0]
    middle = (target_height - tmp_img.shape[0]) // 2

    if padding == "white":
        upper_stack = np.ones(shape=(middle, target_width, channels), dtype=np.uint8)
        lower_stack = np.ones(
            shape=(target_height - height - middle, target_width, channels),
            dtype=np.uint8,
        )

        upper_stack *= 255
        lower_stack *= 255
    else:
        upper_stack = np.zeros(shape=(middle, target_width, channels), dtype=np.uint8)
        lower_stack = np.zeros(
            shape=(target_height - height - middle, target_width, channels),
            dtype=np.uint8,
        )

    out_img = np.vstack([upper_stack, tmp_img, lower_stack])

    return out_img


def pad_to_height(
    img: npt.NDArray, target_width: int, target_height: int, padding: str
) -> npt.NDArray:
    height, _, channels = img.shape
    tmp_img, ratio = resize_to_height(img, target_height)

    width = tmp_img.shape[1]
    middle = (target_width - width) // 2

    if padding == "white":
        left_stack = np.ones(shape=(target_height, middle, channels), dtype=np.uint8)
        right_stack = np.ones(
            shape=(target_height, target_width - width - middle, channels),
            dtype=np.uint8,
        )

        left_stack *= 255
        right_stack *= 255

    else:
        left_stack = np.zeros(shape=(target_height, middle, channels), dtype=np.uint8)
        right_stack = np.zeros(
            shape=(target_height, target_width - width - middle, channels),
            dtype=np.uint8,
        )

    out_img = np.hstack([left_stack, tmp_img, right_stack])

    return out_img


"""
These are basically the two step inbetween the raw line prediction and the OCR pass
"""


def get_line_data(
    image: npt.NDArray, line_mask: npt.NDArray, group_chunks: bool = True
) -> LineData:
    angle = get_rotation_angle_from_lines(line_mask)

    rot_mask = rotate_from_angle(line_mask, angle)
    rot_img = rotate_from_angle(image, angle)

    line_contours = get_contours(rot_mask)
    line_data = [build_line_data(x) for x in line_contours]
    line_data = [x for x in line_data if x.bbox.h > 10]
    sorted_lines, _ = sort_lines_by_threshold2(
        rot_mask, line_data, group_lines=group_chunks
    )

    data = LineData(rot_img, rot_mask, angle, sorted_lines)

    return data


def extract_line_images(
    data: LineData, k_factor: float = 0.75, binarization: bool = True
):
    line_images = [extract_line(x, data.image, k_factor) for x in data.lines]

    if binarization:
        line_images = [binarize(x) for x in line_images]

    return line_images


def get_charset(charset: str) -> List[str]:
    charset = f"ß{charset}"
    return [x for x in charset]


def read_ocr_model_config(config_file: str):
    model_dir = os.path.dirname(config_file)
    file = open(config_file, encoding="utf-8")
    json_content = json.loads(file.read())

    onnx_model_file = f"{model_dir}/{json_content['onnx-model']}"

    input_width = json_content["input_width"]
    input_height = json_content["input_height"]
    input_layer = json_content["input_layer"]
    output_layer = json_content["output_layer"]
    squeeze_channel_dim = (
        True if json_content["squeeze_channel_dim"] == "yes" else False
    )
    swap_hw = True if json_content["swap_hw"] == "yes" else False
    characters = get_charset(json_content["charset"])

    config = OCRConfig(
        onnx_model_file,
        input_width,
        input_height,
        input_layer,
        output_layer,
        squeeze_channel_dim,
        swap_hw,
        characters,
    )

    return config


def read_line_model_config(config_file: str) -> LineDetectionConfig:
    model_dir = os.path.dirname(config_file)
    file = open(config_file, encoding="utf-8")
    json_content = json.loads(file.read())

    onnx_model_file = f"{model_dir}/{json_content['onnx-model']}"
    patch_size = int(json_content["patch_size"])

    config = LineDetectionConfig(onnx_model_file, patch_size)

    return config


def read_layout_model_config(config_file: str) -> LayoutDetectionConfig:
    model_dir = os.path.dirname(config_file)
    file = open(config_file, encoding="utf-8")
    json_content = json.loads(file.read())

    onnx_model_file = f"{model_dir}/{json_content['onnx-model']}"
    patch_size = int(json_content["patch_size"])
    classes = json_content["classes"]

    config = LayoutDetectionConfig(onnx_model_file, patch_size, classes)

    return config
