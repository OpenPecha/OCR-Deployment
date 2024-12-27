import cv2
import pyewts
from pathlib import Path
from Inference import OCRPipeline
from Config import init_monlam_line_model, init_monlam_ocr_model
from Utils import read_line_model_config
from datetime import datetime

pyewt = pyewts.pyewts()

LOG_FILE = "processing_log.txt"


def get_page_unicode(line_texts: list) -> str:
    page_text = ""
    for line_wylie in line_texts:
        line_unicode = pyewt.toUnicode(line_wylie)
        page_text += line_unicode + "\n"
    return page_text


def initialize_ocr_pipeline(ocr_model_name: str, output_dir: str) -> OCRPipeline:
    line_model_config = init_monlam_line_model()
    ocr_config = init_monlam_ocr_model(ocr_model_name)
    line_config = read_line_model_config(line_model_config)
    ocr = OCRPipeline(
        ocr_config=ocr_config,
        line_config=line_config,
        output_dir=output_dir,
    )
    return ocr


def convert_image_to_jpg(image_path: Path, output_dir: Path) -> Path:
    image = cv2.imread(str(image_path))
    image_name = image_path.stem + ".jpg"
    output_image_path = output_dir / image_name
    cv2.imwrite(str(output_image_path), image)

    return output_image_path


def log_processed_image(image_name: str) -> None:
    with open(LOG_FILE, "a") as log_file:
        log_file.write(f"{datetime.now()} - Processed: {image_name}\n")


def has_been_processed(image_name: str) -> bool:
    if Path(LOG_FILE).exists():
        with open(LOG_FILE, "r") as log_file:
            log_lines = log_file.readlines()
            for line in log_lines:
                if image_name in line:
                    return True
    return False


def process_image(ocr: OCRPipeline, image_path: Path, output_dir: Path) -> None:
    image_name = image_path.stem
    if has_been_processed(image_name):
        print(f"Skipping already processed image: {image_name}")
        return

    image = cv2.imread(str(image_path))

    page_text, line_data, line_images = ocr.run_ocr(image=image, image_name=image_name)

    # Save Unicode text
    unicode_text = get_page_unicode(page_text)
    text_dir = output_dir / "text"
    text_dir.mkdir(parents=True, exist_ok=True)
    output_file = text_dir / f"{image_name}.txt"
    output_file.write_text(unicode_text, encoding="utf-8")

    # Save line images
    line_images_dir = output_dir / "line_images"
    line_images_dir.mkdir(parents=True, exist_ok=True)
    ocr.save_line_images(line_images, str(line_images_dir))

    log_processed_image(image_name)

    print(f"text saved to {output_file} and line images saved to {line_images_dir}")


def process_directory(input_dir: Path, output_dir: Path, ocr_model_name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    ocr = initialize_ocr_pipeline(ocr_model_name, str(output_dir))
    for subfolder in input_dir.iterdir():
        if subfolder.is_dir():
            for image_path in subfolder.glob("*"): 
                if image_path.suffix.lower() not in [".jpg"]:
                    image_path = convert_image_to_jpg(image_path, subfolder)
                process_image(ocr, image_path, output_dir)


def main():
    input_dir = Path("/Users/tenkal/OpenPecha/ocr-e2e-benchmark/data/source_images")
    output_dir = Path("data/output")
    ocr_model_name = "Woodblock"

    process_directory(input_dir, output_dir, ocr_model_name)


if __name__ == "__main__":
    main()
