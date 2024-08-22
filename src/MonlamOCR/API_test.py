import requests

def send_request_to_api(endpoint, data=None, method="POST"):
    base_url = "http://18.206.160.30:8000"
    url = f"{base_url}{endpoint}"

    try:
        response = requests.post(url, json=data)
        return response.json()['unicode_text']
    except requests.exceptions.RequestException as e:
        return {"error": "An error occurred while making the request", "details": str(e)}


# For a POST request with JSON data:
data = {
    "image_url": "https://s3.amazonaws.com/monlam.ai.ocr/Test/input/I8LS169351039.jpg"
}
Transcript = send_request_to_api("/process", data=data, method="POST")
print(Transcript)
