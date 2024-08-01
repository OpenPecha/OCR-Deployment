"""
run e.g. with: python run_xml_export.py -i "Data/W30337" -e "tif"
"""

import os
import cv2
import sys
import argparse
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from MonlamOCR.Exporter import PageXMLExporter
from MonlamOCR.Inference import LineDetection, OCRPipeline
from MonlamOCR.Utils import (
    get_text_bbox,
    get_file_name,
    create_dir,
    get_line_data,
    read_line_model_config,
)

LINE_MODEL = "MonlamOCR/Models/Lines/config.json"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, required=False, default="page")
    parser.add_argument(
        "-e", "--file_extension", type=str, required=False, default="jpg"
    )
    parser.add_argument("-k", "--k_factor", type=float, required=False, default=0.75)

    args = parser.parse_args()
    input_dir = args.input
    
    file_ext = args.file_extension
    k_factor = args.k_factor

    if not os.path.isdir(input_dir):
        print(f"ERROR: Input dir is not a valid directory")
        sys.exit(1)

    images = natsorted(glob(f"{input_dir}/*.{file_ext}"))
    print(f"Images: {len(images)}")

    if len(images) > 0:
        output_dir = os.path.join(input_dir, args.output)
        create_dir(output_dir)

        line_model_config = read_line_model_config(LINE_MODEL)
        line_infernce = LineDetection(line_model_config)
        xml_exporter = PageXMLExporter()

        for idx, image_path in tqdm(enumerate(images), total=len(images)):
            img = cv2.imread(image_path)
            img_name = get_file_name(image_path)
            line_mask = line_infernce.predict(img, fix_height=True)
            line_data = get_line_data(img, line_mask)

            if len(line_data.lines) > 0:
                text_bbox = get_text_bbox(line_data)

                xml_doc = xml_exporter.build_xml_document(
                    line_data.image,
                    img_name,
                    [],
                    text_bbox,
                    line_data.lines,
                    [],
                    [],
                    [],
                    line_data.angle,
                )

                out_file = f"{output_dir}/{img_name}.xml"
                with open(out_file, "w", encoding="UTF-8") as f:
                    f.write(xml_doc)