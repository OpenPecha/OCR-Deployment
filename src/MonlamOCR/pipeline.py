import cv2
import pyewts
from pathlib import Path
from Inference import OCRPipeline
from Config import init_monlam_line_model, init_monlam_ocr_model
from Utils import read_line_model_config
pyewt = pyewts.pyewts()


def get_page_unicode(line_texts: list) -> str:
    """
    Converts a list of Wylie transliterations to Unicode.

    Args:
        line_texts (list): List of Wylie transliterations.

    Returns:
        str: Combined Unicode text with lines separated by newlines.
    """
    page_text = ""
    for line_wylie in line_texts:
        line_unicode = pyewt.toUnicode(line_wylie)
        page_text += line_unicode + "\n"
    return page_text


def initialize_ocr_pipeline(ocr_model_name: str, output_dir: str) -> OCRPipeline:
    """
    Initializes the OCR pipeline with the given model configuration.

    Args:
        ocr_model_name (str): The name of the OCR model to use.
        output_dir (str): Path to the output directory.

    Returns:
        OCRPipeline: Initialized OCR pipeline object.
    """
    line_model_config = init_monlam_line_model()
    ocr_config = init_monlam_ocr_model(ocr_model_name)
    line_config = read_line_model_config(line_model_config)
    ocr = OCRPipeline(
        ocr_config=ocr_config,
        line_config=line_config,
        output_dir=output_dir,
    )
    return ocr


def process_image(ocr: OCRPipeline, image_path: Path, output_dir: Path) -> None:
    """
    Processes a single image and saves its OCR output.

    Args:
        ocr (OCRPipeline): Initialized OCR pipeline.
        image_path (Path): Path to the input image.
        output_dir (Path): Path to the output directory.
    """
    image_name = image_path.stem
    image = cv2.imread(str(image_path))

    page_text, line_data, line_text = ocr.run_ocr(image=image)

    unicode_text = get_page_unicode(page_text)

    output_file = output_dir / f"{image_name}.txt"
    output_file.write_text(unicode_text, encoding="utf-8")
    print(f"Output saved to {output_file}")


def process_directory(input_dir: Path, output_dir: Path, ocr_model_name: str) -> None:
    """
    Processes all images in a directory.

    Args:
        input_dir (Path): Path to the directory containing input images.
        output_dir (Path): Path to the directory to save OCR results.
        ocr_model_name (str): The name of the OCR model to use.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    ocr = initialize_ocr_pipeline(ocr_model_name, str(output_dir))
    for image_path in input_dir.glob("*.jpg"):
        process_image(ocr, image_path, output_dir)


def main():
    input_dir = Path("data/input")
    output_dir = Path("data/output")
    ocr_model_name = "Woodblock"

    process_directory(input_dir, output_dir, ocr_model_name)


if __name__ == "__main__":
    main()
