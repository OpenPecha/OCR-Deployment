import requests
import cv2
import numpy as np
from pathlib import Path


def send_request_to_api(url, data):
    try:
        response = requests.post(url, json=data)
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": "An error occurred while making the request", "details": str(e)}


if __name__ == "__main__":
    data = {
        "image_url": "https://s3.amazonaws.com/monlam.ai.ocr/Test/input/0001.png",
        "OCR_model": "Woodblock"
    }
    response = send_request_to_api("http://localhost:8000/process/", data)
    print(response)


# Some example images to test the API
# https://s3.amazonaws.com/monlam.ai.ocr/Test/input/0001.png
# https://s3.amazonaws.com/monlam.ai.ocr/Test/input/0002.png
# https://s3.amazonaws.com/monlam.ai.ocr/Test/input/0003.png
# https://s3.amazonaws.com/monlam.ai.ocr/Test/input/I3CN78390293.png
