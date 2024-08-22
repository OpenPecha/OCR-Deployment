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
from natsort import natsorted
from MonlamOCR.Utils import create_dir, get_file_name, read_line_model_config, read_layout_model_config
from MonlamOCR.Config import init_monlam_line_model, init_monlam_layout_model, init_monla_ocr_model
from MonlamOCR.Inference import OCRPipeline
from MonlamOCR.Exporter import PageXMLExporter, JsonExporter


line_model_config_file = init_monlam_line_model()
layout_model_config_file = init_monlam_layout_model()

line_model_config = read_line_model_config(line_model_config_file)
layout_model_config = read_layout_model_config(layout_model_config_file)

ocr_model_config = init_monla_ocr_model("Woodblock")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, required=False, default="Output")
    parser.add_argument("-e", "--file_extension", type=str, required=False, default="jpg")
    parser.add_argument("-m", "--mode", choices=["Line", "Layout"], default="Line")
    parser.add_argument("-k", "--k_factor", type=float, required=False, default=1.2)
    parser.add_argument("-s", "--save_format", choices=["xml", "json"], default="xml")

    args = parser.parse_args()
    input_dir = args.input
    output_dir = args.output
    file_ext = args.file_extension
    mode = args.mode
    k_factor = args.k_factor
    save_format = args.save_format

    if not os.path.isdir(input_dir):
        print(f"ERROR: Input dir is not a valid directory")
        sys.exit(1)
    
    images = natsorted(glob(f"{input_dir}/*.{file_ext}"))
    print(f"Images: {len(images)}")

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
        # export the text directly as text file or turn export off and do other stuff with the text
        # line information is stored in the returned line_data
        image_name = get_file_name(image_path)
        image = cv2.imread(image_path)
        text, line_data, line_images = ocr_pipeline.run_ocr(image)

        text = [converter.toUnicode(x) for x in text]
        exporter.export_lines(image, image_name, line_data, text_lines=text)
        
