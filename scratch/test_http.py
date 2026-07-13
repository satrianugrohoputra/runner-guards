import requests

try:
    response = requests.get("http://localhost:8501/")
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        print("Streamlit app is serving successfully!")
    else:
        print("Failed to serve app.")
except Exception as e:
    print(f"Error: {e}")
