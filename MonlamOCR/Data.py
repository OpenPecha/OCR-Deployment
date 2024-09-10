import numpy.typing as npt
from typing import Tuple, List, Dict
from dataclasses import dataclass


@dataclass
class BBox:
    x: int
    y: int
    w: int
    h: int


@dataclass
class Line:
    contour: npt.NDArray
    bbox: BBox
    center: Tuple[int, int]


@dataclass
class LineData:
    image: npt.NDArray
    prediction: npt.NDArray
    angle: float
    lines: List[Line]


@dataclass
class LayoutData:
    image: npt.NDArray
    rotation: float
    images: List[BBox]
    text_bboxes: List[BBox]
    lines: List[Line]
    captions: List[BBox]
    margins: List[BBox]
    predictions: Dict[str, npt.NDArray]


@dataclass
class LineDetectionConfig:
    model_file: str
    patch_size: int


@dataclass
class LayoutDetectionConfig:
    model_file: str
    patch_size: int
    classes: List[str]


@dataclass
class OCRConfig:
    model_file: str
    input_width: int
    input_height: int
    input_layer: str
    output_layer: str
    squeeze_channel: bool
    swap_hw: bool
    charset: List[str]
