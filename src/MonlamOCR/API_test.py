import requests

def send_request_to_api(url, data):
    try:
        response = requests.post(url, json=data)
        return response.json()['unicode_text']
    except requests.exceptions.RequestException as e:
        return {"error": "An error occurred while making the request", "details": str(e)}


data = {
    "image_url": "https://s3.amazonaws.com/monlam.ai.ocr/Test/input/I8LS169351039.jpg"
}
Transcript = send_request_to_api("http://18.206.160.30:8000/process", data)
print(Transcript)



# Some example images to test the API 
# https://s3.amazonaws.com/monlam.ai.ocr/Test/input/0001.png
# https://s3.amazonaws.com/monlam.ai.ocr/Test/input/0002.png
# https://s3.amazonaws.com/monlam.ai.ocr/Test/input/0003.png
# https://s3.amazonaws.com/monlam.ai.ocr/Test/input/I3CN78390293.png