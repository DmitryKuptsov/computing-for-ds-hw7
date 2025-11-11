import requests
from requests.exceptions import RequestException, Timeout, ConnectionError, HTTPError

API_URL = "http://127.0.0.1:8000/predict"
JSON_FILE = "sample_json.json"

try:
    with open(JSON_FILE, "rb") as f:
        files = {"file": (JSON_FILE, f, "application/json")}
        response = requests.post(API_URL, files=files, timeout=10)
        response.raise_for_status()

        print("Prediction:", response.json())

except HTTPError as e:
    print(f"HTTP error: {e}")
except (RequestException, Timeout, ConnectionError) as e:
    print(f"Request failed: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
