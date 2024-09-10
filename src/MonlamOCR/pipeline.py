import cv2
import pyewts
from pathlib import Path
from MonlamOCR.Inference import OCRPipeline
from MonlamOCR.Config import init_monlam_line_model, init_monla_ocr_model
from MonlamOCR.Utils import read_line_model_config, read_ocr_model_config


pyewt = pyewts.pyewts()

def get_page_unicode(line_texts: list) -> str:
    page_text = ""
    for line_wylie in line_texts:
        line_unicode = pyewt.toUnicode(line_wylie)
        page_text += line_unicode + "\n"
    return page_text


def get_line_data_and_page_text(line_inference, image_name):
    line_data = []
    page_text = ""
    for line_num, line_info in enumerate(line_inference, 1):
        line_dict = {
            "line_id": f"{image_name}_{line_num}",
            "line_text": pyewt.toUnicode(line_info['text']) ,
            "line_annotation": line_info['line_annotation']
        }
        line_data.append(line_dict)
        page_text += pyewt.toUnicode(line_info['text']) + "\n"
    return line_data, page_text



def main():
    line_model_config = init_monlam_line_model()
    ocr_config = init_monla_ocr_model("GoogleBooks_E")
    line_config = read_line_model_config(line_model_config)
    ocr = OCRPipeline(
        ocr_config=ocr_config,
        line_config=line_config,
        output_dir="./data/output",
    )
    image_paths = [
        "/Users/tashitsering/Desktop/Work/OCR-Deployment/data/00-3.jpg",
        "./data/0003.jpg",
        "./data/test_images/0002.jpg",
        "./data/test_images/0003.jpg",
        "./data/test_images/0004.png",
        "./data/test_images/0005.png",
    ]
    for image_path in image_paths:
        image_name = image_path.split("/")[-1].split(".")[0]
        image = cv2.imread(image_path)
        line_inference = ocr.run_ocr(
            image=image

        )

        line_data, page_text = get_line_data_and_page_text(line_inference, image_name)
        # text = get_page_unicode(page_text)
        # Path(f"./data/output/{image_name}.txt").write_text(text)

    



if __name__ == "__main__":
    main()
    
