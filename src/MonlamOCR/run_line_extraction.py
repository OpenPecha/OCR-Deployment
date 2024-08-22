"""
run e.g. with: 
- python run_line_extraction.py -i "SampleData" -e "jpg"

or using the layout model:
- python run_line_extraction.py -i "SampleData" -e "jpg" -m "Layout"
"""

import os
import cv2
import sys
import argparse
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from MonlamOCR.Utils import create_dir, extract_line_images, get_file_name, get_line_data, read_line_model_config, read_layout_model_config, binarize
from MonlamOCR.Config import init_monlam_line_model, init_monlam_layout_model
from MonlamOCR.Inference import LineDetection, LayoutDetection



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, required=False, default="Output")
    parser.add_argument("-e", "--file_extension", type=str, required=False, default="jpg")
    parser.add_argument("-m", "--mode", choices=["Line", "Layout"], default="Line")
    parser.add_argument("-k", "--k_factor", type=float, required=False, default=1.2)

    args = parser.parse_args()
    input_dir = args.input
    output_dir = args.output
    file_ext = args.file_extension
    mode = args.mode
    k_factor = args.k_factor

    if not os.path.isdir(input_dir):
        print(f"ERROR: Input dir is not a valid directory")
        sys.exit(1)
    
    images = natsorted(glob(f"{input_dir}/*.{file_ext}"))
    print(f"Images: {len(images)}")

    create_dir(output_dir)

    if mode == "Line":
        line_model_config_file = init_monlam_line_model()
        line_model_config = read_line_model_config(line_model_config_file)
        line_inference = LineDetection(line_model_config)

        for idx, image_path in tqdm(enumerate(images), total=len(images)):
            image_n = get_file_name(image_path)
            image = cv2.imread(image_path)
            image = binarize(image)

            line_mask = line_inference.predict(image, fix_height=True)
            line_data = get_line_data(image, line_mask)
            line_images = extract_line_images(line_data, k_factor)

            for l_idx, line_img in enumerate(line_images):
                out_file = f"{output_dir}/{image_n}_{l_idx}.jpg"
                cv2.imwrite(out_file, line_img)

    else:
        layout_model_config_file = init_monlam_layout_model()
        layout_model_config = read_layout_model_config(layout_model_config_file)
        layout_inference = LayoutDetection(layout_model_config)

        for idx, image_path in tqdm(enumerate(images), total=len(images)):
            image_n = get_file_name(image_path)
            image = cv2.imread(image_path)
            image = binarize(image)

            layout_mask = layout_inference.predict(image)
            line_data = get_line_data(
                image, layout_mask[:, :, 2]
            )  # for the dim, see classes in the layout config file
            line_images = extract_line_images(line_data, k_factor)

            for l_idx, line_img in enumerate(line_images):
                out_file = f"{output_dir}/{image_n}_{l_idx}.jpg"
                cv2.imwrite(out_file, line_img)
