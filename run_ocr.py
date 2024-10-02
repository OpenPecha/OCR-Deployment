"""
run e.g. with: 
- python run_ocr.py -i "SampleData" -e "jpg"

or using the layout model:
- python run_ocr.py -i "SampleData" -e "jpg" -m "Layout"
"""

import os
import cv2
import sys
import pyewts
import argparse
from glob import glob
from tqdm import tqdm
from MonlamOCR.Data import OCRStatus
from natsort import natsorted
from MonlamOCR.Utils import create_dir, get_file_name, read_line_model_config, read_layout_model_config
from MonlamOCR.Config import init_monlam_line_model, init_monlam_layout_model, init_monlam_ocr_model
from MonlamOCR.Inference import OCRPipeline
from MonlamOCR.Exporter import PageXMLExporter, JsonExporter


line_model_config_file = init_monlam_line_model()
layout_model_config_file = init_monlam_layout_model()

line_model_config = read_line_model_config(line_model_config_file)
layout_model_config = read_layout_model_config(layout_model_config_file)

ocr_model_config = init_monlam_ocr_model("Woodblock")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, required=False, default="Output")
    parser.add_argument("-e", "--file_extension", type=str, required=False, default="jpg")
    parser.add_argument("-m", "--mode", choices=["Line", "Layout"], default="Layout")
    parser.add_argument("-k", "--k_factor", type=float, required=False, default=1.7)
    parser.add_argument("-s", "--save_format", choices=["xml", "json"], default="xml")

    args = parser.parse_args()
    input_dir = args.input
    output_dir = args.output
    file_ext = args.file_extension
    mode = args.mode
    k_factor = args.k_factor
    save_format = args.save_format

    if not os.path.isdir(input_dir):
        print("ERROR: Input dir is not a valid directory")
        sys.exit(1)
    
    images = natsorted(glob(f"{input_dir}/*.{file_ext}"))
    print(f"Images: {len(images)}")


    dir_name = os.path.basename(input_dir)
    output_dir = os.path.join(output_dir, dir_name)
    create_dir(output_dir)

    converter = pyewts.pyewts()

    if save_format == "xml":
        exporter = PageXMLExporter(output_dir)
    else:
        exporter = JsonExporter(output_dir)

    if mode == "Line":
        ocr_pipeline = OCRPipeline(ocr_model_config, line_model_config, output_dir)

    else:
        ocr_pipeline = OCRPipeline(ocr_model_config, layout_model_config, output_dir)

    for idx, image_path in tqdm(enumerate(images), total=len(images)):
        image_name = get_file_name(image_path)
        img = cv2.imread(image_path)
        status, ocr_result = ocr_pipeline.run_ocr(img, image_name)

        if status == OCRStatus.SUCCESS:
            if len(ocr_result.lines) > 0:
                text = [converter.toUnicode(x) for x in ocr_result.text]
                exporter.export_lines(img, image_name, ocr_result.lines, text, angle=ocr_result.angle)
