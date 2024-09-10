import os
from typing import List, Tuple
import cv2
import numpy as np
import numpy.typing as npt
import onnxruntime as ort
from scipy.special import softmax
from MonlamOCR.Config import LAYOUT_COLORS
from MonlamOCR.Data import (
    LineData,
    OCRConfig,
    LineDetectionConfig,
    LayoutDetectionConfig,
)
from pyctcdecode import build_ctcdecoder
from MonlamOCR.Utils import (
    create_dir,
    extract_line_images,
    preprocess_image,
    get_file_name,
    binarize,
    get_line_data,
    normalize,
    stitch_predictions,
    tile_image,
    sigmoid,
    resize_to_height,
    pad_to_height,
    pad_to_width,
    read_ocr_model_config,
)


class Detection:
    def __init__(self, config: LineDetectionConfig | LayoutDetectionConfig):
        self.config = config
        self._config_file = config
        self._onnx_model_file = config.model_file
        self._patch_size = config.patch_size
        self._execution_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._inference = ort.InferenceSession(
            self._onnx_model_file, providers=self._execution_providers
        )

    def _preprocess_image(self, image: npt.NDArray, patch_size: int = 512):
        padded_img, pad_x, pad_y = preprocess_image(image, patch_size)
        tiles, y_steps = tile_image(padded_img, patch_size)
        tiles = [binarize(x) for x in tiles]
        tiles = [normalize(x) for x in tiles]
        tiles = np.array(tiles)

        return padded_img, tiles, y_steps, pad_x, pad_y

    def _crop_prediction(
        self, image: npt.NDArray, prediction: npt.NDArray, x_pad: int, y_pad: int
    ) -> npt.NDArray:
        x_lim = prediction.shape[1] - x_pad
        y_lim = prediction.shape[0] - y_pad

        prediction = prediction[:y_lim, :x_lim]
        prediction = cv2.resize(prediction, dsize=(image.shape[1], image.shape[0]))

        return prediction

    def _predict(self, image_batch: npt.NDArray):
        image_batch = np.transpose(image_batch, axes=[0, 3, 1, 2])
        ort_batch = ort.OrtValue.ortvalue_from_numpy(image_batch)
        prediction = self._inference.run_with_ort_values(
            ["output"], {"input": ort_batch}
        )
        prediction = prediction[0].numpy()

        return prediction

    def predict(self, image: npt.NDArray, class_threshold: float = 0.8) -> npt.NDArray:
        pass


class LineDetection(Detection):
    def __init__(self, config: LineDetectionConfig) -> None:
        super().__init__(config)

    def predict(self, image: npt.NDArray, class_threshold: float = 0.9) -> npt.NDArray:

        _, tiles, y_steps, pad_x, pad_y = self._preprocess_image(
            image, patch_size=self._patch_size
        )
        prediction = self._predict(tiles)
        prediction = np.squeeze(prediction, axis=1)
        prediction = sigmoid(prediction)
        prediction = np.where(prediction > class_threshold, 1.0, 0.0)
        merged_image = stitch_predictions(prediction, y_steps=y_steps)
        merged_image = self._crop_prediction(image, merged_image, pad_x, pad_y)
        merged_image = merged_image.astype(np.uint8)
        merged_image *= 255

        return merged_image


class LayoutDetection(Detection):
    def __init__(self, config: LayoutDetectionConfig) -> None:
        super().__init__(config)
        self._classes = config.classes
        print(f"Layout Classes: {self._classes}")

    def create_preview_image(
        image: npt.NDArray,
        image_predictions: list,
        line_predictions: list,
        caption_predictions: list,
        margin_predictions: list,
        alpha: float = 0.4,
    ) -> npt.NDArray:
        
        if image is None:
            return None
        
        mask = np.zeros(image.shape, dtype=np.uint8)

        if len(image_predictions) > 0:
            color = tuple([int(x) for x in LAYOUT_COLORS["image"].split(",")])

            for idx, _ in enumerate(image_predictions):
                cv2.drawContours(
                    mask, image_predictions, contourIdx=idx, color=color, thickness=-1
                )

        if len(line_predictions) > 0:
            color = tuple([int(x) for x in LAYOUT_COLORS["line"].split(",")])

            for idx, _ in enumerate(line_predictions):
                cv2.drawContours(
                    mask, line_predictions, contourIdx=idx, color=color, thickness=-1
                )

        if len(caption_predictions) > 0:
            color = tuple([int(x) for x in LAYOUT_COLORS["caption"].split(",")])

            for idx, _ in enumerate(caption_predictions):
                cv2.drawContours(
                    mask, caption_predictions, contourIdx=idx, color=color, thickness=-1
                )

        if len(margin_predictions) > 0:
            color = tuple([int(x) for x in LAYOUT_COLORS["margin"].split(",")])

            for idx, _ in enumerate(margin_predictions):
                cv2.drawContours(
                    mask, margin_predictions, contourIdx=idx, color=color, thickness=-1
                )

        cv2.addWeighted(mask, alpha, image, 1 - alpha, 0, image)

        return image

    def predict(self, image: npt.NDArray, class_threshold: float = 0.8) -> npt.NDArray:
        _, tiles, y_steps, pad_x, pad_y = self._preprocess_image(
            image, patch_size=self._patch_size
        )
        prediction = self._predict(tiles)
        prediction = np.transpose(prediction, axes=[0, 2, 3, 1])
        prediction = softmax(prediction, axis=-1)
        prediction = np.where(prediction > class_threshold, 1.0, 0)
        merged_image = stitch_predictions(prediction, y_steps=y_steps)
        merged_image = self._crop_prediction(image, merged_image, pad_x, pad_y)
        merged_image = merged_image.astype(np.uint8)
        merged_image *= 255

        return merged_image


class OCRInference:
    def __init__(self, ocr_config: OCRConfig) -> None:
        self.config = ocr_config
        self._onnx_model_file = ocr_config.model_file
        self._input_width = ocr_config.input_width
        self._input_height = ocr_config.input_height
        self._input_layer = ocr_config.input_layer
        self._output_layer = ocr_config.output_layer
        self._characters = ocr_config.charset
        self._squeeze_channel_dim = ocr_config.squeeze_channel
        self._swap_hw = ocr_config.swap_hw
        self._ctcdecoder = build_ctcdecoder(self._characters)
        self._execution_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.ocr_session = ort.InferenceSession(
            self._onnx_model_file, providers=self._execution_providers
        )

    def _pad_ocr_line(
        self,
        img: npt.NDArray,
        padding: str = "black",
    ) -> npt.NDArray:

        width_ratio = self._input_width / img.shape[1]
        height_ratio = self._input_height / img.shape[0]

        if width_ratio < height_ratio:
            out_img = pad_to_width(img, self._input_width, self._input_height, padding)

        elif width_ratio > height_ratio:
            out_img = pad_to_height(img, self._input_width, self._input_height, padding)
        else:
            out_img = pad_to_width(img, self._input_width, self._input_height, padding)

        return cv2.resize(
            out_img,
            (self._input_width, self._input_height),
            interpolation=cv2.INTER_LINEAR,
        )

    def _prepare_ocr_line(self, image: npt.NDArray) -> npt.NDArray:
        line_image = self._pad_ocr_line(image)
        line_image = cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY)
        line_image = line_image.reshape((1, self._input_height, self._input_width))
        line_image = (line_image / 127.5) - 1.0
        line_image = line_image.astype(np.float32)

        return line_image

    def _predict(self, image_batch: npt.NDArray) -> npt.NDArray:
        image_batch = image_batch.astype(np.float32)
        ort_batch = ort.OrtValue.ortvalue_from_numpy(image_batch)
        ocr_results = self.ocr_session.run_with_ort_values(
            [self._output_layer], {self._input_layer: ort_batch}
        )

        logits = ocr_results[0].numpy()
        logits = np.squeeze(logits)

        return logits

    def _decode(self, logits: npt.NDArray) -> str:

        if logits.shape[0] == len(self._characters):
            logits = np.transpose(
                logits, axes=[1, 0]
            )  # adjust logits to have shape time, vocab

        text = self._ctcdecoder.decode(logits)
        text = text.replace(" ", "")
        text = text.replace("ß", "")
        text = text.replace("§", " ")

        return text

    def run(self, line_image: npt.NDArray) -> str:
        line_image = self._prepare_ocr_line(line_image)

        if self._swap_hw:
            line_image = np.transpose(line_image, axes=[0, 2, 1])

        if not self._squeeze_channel_dim:
            line_image = np.expand_dims(line_image, axis=1)

        logits = self._predict(line_image)
        text = self._decode(logits)

        return text


class OCRPipeline:
    """
    Note: The handling of line model vs. layout model is kind of provisional here and totally depends on the way you want to run this.
    You could also pass both configs to the the pipeline, run both models and merge the (partially) overlapping output before extracting the line images to compensate for the strengths/weaknesses
    of either model. So that is basically up to you.

    """

    def __init__(
        self,
        ocr_config: str,
        line_config: LineDetectionConfig | LayoutDetectionConfig,
        output_dir: str,
    ):
        self.ready = False
        self.ocr_model_config = read_ocr_model_config(ocr_config)
        self.line_config = line_config
        self.ocr_inference = OCRInference(self.ocr_model_config)

        if isinstance(self.line_config, LineDetectionConfig):
            print(f"Running OCR in Line Mode")
            self.line_inference = LineDetection(self.line_config)
            self.ready = True
        elif isinstance(self.line_config, LayoutDetectionConfig):
            print(f"Running OCR in Layout Mode")
            self.line_inference = LayoutDetection(self.line_config)
            self.ready = True
        else:
            self.line_inference = None
            self.ready = False

        if not os.path.isdir(output_dir):
            create_dir(output_dir)

        self.output_dir = output_dir

    def _predict(
        self, image: npt.NDArray, k_factor: float
    ) -> Tuple[List[str], LineData, List[npt.NDArray]]:

        if isinstance(self.line_config, LineDetectionConfig):
            line_mask = self.line_inference.predict(image)
            line_data = get_line_data(image, line_mask)
            line_images = extract_line_images(line_data, k_factor)
        else:
            layout_mask = self.line_inference.predict(image)
            line_data = get_line_data(
                image, layout_mask[:, :, 2]
            )  # for the dim, see classes in the layout config file
            line_images = extract_line_images(line_data, k_factor)

        page_text = []
        filtered_lines = []

        for line_img, line_info in zip(line_images, line_data.lines):
            pred = self.ocr_inference.run(line_img)
            pred = pred.strip()

            if pred != "":
                page_text.append(pred)
                filtered_lines.append(line_info)

        filtered_line_data = LineData(
            line_data.image, line_data.prediction, line_data.angle, filtered_lines
        )

        return page_text, filtered_line_data, line_images

    def run_ocr(
        self, image: npt.NDArray, k_factor: float = 1.2
    ) -> Tuple[List[str], LineData, List[npt.NDArray]]:
        page_text, line_data, line_images = self._predict(image, k_factor)

        return page_text, line_data, line_images
