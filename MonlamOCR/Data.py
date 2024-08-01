import numpy.typing as npt
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
    center: tuple[int, int]

@dataclass
class LineDetectionConfig:
    model_file: str
    patch_size: int

@dataclass
class OCRConfig:
    model_file: str
    input_width: int
    input_height: int
    input_layer: str
    output_layer: str
    squeeze_channel: bool
    swap_hw: bool
    charset: list[str]
