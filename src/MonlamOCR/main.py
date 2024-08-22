from fastapi import FastAPI, HTTPException
import logging
import cv2
import os
import tempfile
import pyewts
import requests
from MonlamOCR.Inference import OCRPipeline
from MonlamOCR.Config import init_monlam_line_model, init_monla_ocr_model
from MonlamOCR.Utils import read_line_model_config

app = FastAPI()

# Initialize your models and configs
pyewt = pyewts.pyewts()

line_model_config = init_monlam_line_model()
ocr_config = init_monla_ocr_model("Woodblock")
line_config = read_line_model_config(line_model_config)
ocr_pipeline = OCRPipeline(
    ocr_config=ocr_config,
    line_config=line_config,
    output_dir="./data/output",
)

def get_page_unicode(line_texts: list) -> str:
    page_text = ""
    for line_wylie in line_texts:
        line_unicode = pyewt.toUnicode(line_wylie)
        page_text += line_unicode + "\n"
    return page_text

def download_image(image_url: str) -> str:
    try:
        # Send a GET request to fetch the image from the URL
        response = requests.get(image_url)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx and 5xx)

        # Create a temporary file to save the image
        temp_dir = tempfile.gettempdir()
        temp_file_path = os.path.join(temp_dir, "temp_image.jpg")

        # Write the image content to the temporary file
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(response.content)

        # Check if the image can be opened
        image = cv2.imread(temp_file_path)
        if image is None:
            raise ValueError("Could not decode the image from the URL")

        # Return the path to the saved image
        return temp_file_path

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download image: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode image: {str(e)}")

@app.post("/process/")
async def process_file(data: dict):
    image_url = data['image_url']
    try:

        # Download the image from the URL
        image_path = download_image(image_url)

        # Read the image using OpenCV
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Could not read the image from the saved path.")

        # Run the OCR pipeline
        page_text, line_data, line_text = ocr_pipeline.run_ocr(image=image)

        # Convert to Unicode
        text = get_page_unicode(page_text)

        return {"unicode_text": text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the file: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
