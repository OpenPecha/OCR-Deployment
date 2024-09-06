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



def crop_line_using_contour(image_path, contour):
    """
    Crops a line from the image using the contour to create a mask.

    Parameters:
        image_path (str): Path to the input image.
        contour (list): A list of contour points.

    Returns:
        cropped_image (numpy.ndarray): The cropped image array.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError("The image file was not found.")

    # Convert list of points to a numpy array
    contour_array = np.array(contour, dtype=np.int32)

    # Create a mask with the same dimensions as the image
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Draw the contour on the mask with white color
    cv2.drawContours(mask, [contour_array], -1, (255), thickness=cv2.FILLED)

    # Apply the mask to the image
    cropped_image = cv2.bitwise_and(image, image, mask=mask)

    return cropped_image




if __name__ == "__main__":
    data = {
    "image_url": "https://s3.amazonaws.com/monlam.ai.ocr/Test/input/0001.png",
    "OCR_model": "Woodblock"
    }
    # response = send_request_to_api("http://18.206.160.30:8000/process", data)
    response = send_request_to_api("http://localhost:8000/process/", data=data)
    contour = response['line_data'][0]['line_annotation']['contour']
    image_path = Path(f"./0001.png")
    cropped_image = crop_line_using_contour(image_path, contour)
    save_path = Path(f"./cropped_line.png")
    cv2.imwrite(str(save_path), cropped_image)


# Some example images to test the API 
# https://s3.amazonaws.com/monlam.ai.ocr/Test/input/0001.png
# https://s3.amazonaws.com/monlam.ai.ocr/Test/input/0002.png
# https://s3.amazonaws.com/monlam.ai.ocr/Test/input/0003.png
# https://s3.amazonaws.com/monlam.ai.ocr/Test/input/I3CN78390293.png