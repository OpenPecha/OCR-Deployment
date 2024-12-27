import cv2
import pyewts
from pathlib import Path
from Inference import OCRPipeline
from Config import init_monlam_line_model, init_monlam_ocr_model
from Utils import read_line_model_config
pyewt = pyewts.pyewts()


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


def process_image(ocr: OCRPipeline, image_path: Path, output_dir: Path) -> None:
    image_name = image_path.stem
    image = cv2.imread(str(image_path))

    page_text, line_data, line_images = ocr.run_ocr(image=image, image_name=image_name)


    unicode_text = get_page_unicode(page_text)
    text_dir = output_dir / "text"
    text_dir.mkdir(parents=True, exist_ok=True)  
    output_file = text_dir / f"{image_name}.txt"
    output_file.write_text(unicode_text, encoding="utf-8")

    # Save line images
    line_images_dir = output_dir / "line_images"
    line_images_dir.mkdir(parents=True, exist_ok=True)
    ocr.save_line_images(line_images, str(line_images_dir))

    print(f"Output saved to {output_file} and line images saved to {line_images_dir}")



def process_directory(input_dir: Path, output_dir: Path, ocr_model_name: str) -> None:
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
