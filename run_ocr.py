"""
run e.g. with: python run_ocr.py -i "Data/W30337" -e "tif"
"""

import os
import sys
import argparse
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from MonlamOCR.Inference import OCRPipeline

LINE_MODEL = "MonlamOCR/Models/Lines/config.json"
OCR_MODEL = "MonlamOCR/Models/OCR/Woodblock/config.json"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, required=False, default="Output")
    parser.add_argument("-e", "--file_extension", type=str, required=False, default="jpg")
    parser.add_argument("-k", "--k_factor", type=float, required=False, default=0.75)

    args = parser.parse_args()
    input_dir = args.input
    output_dir = args.output
    file_ext = args.file_extension
    k_factor = args.k_factor

    if not os.path.isdir(input_dir):
        print(f"ERROR: Input dir is not a valid directory")
        sys.exit(1)
    
    images = natsorted(glob(f"{input_dir}/*.{file_ext}"))
    print(f"Images: {len(images)}")

    ocr_pipeline = OCRPipeline(OCR_MODEL, LINE_MODEL, output_dir)

    for idx, image_path in tqdm(enumerate(images), total=len(images)):
        # export the text directly as text file or turn export off and do other stuff with the text
        # line information is stored in the returned line_data
        text, line_data = ocr_pipeline.run_ocr(images[idx], export=True)
