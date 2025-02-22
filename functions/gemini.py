import requests
import os
from dotenv import load_dotenv
import json

load_dotenv()
apiKey = os.getenv("HKBU_API_KEY")
basicUrl = os.getenv("HKBU_BASIC_URL")
if apiKey is None:
    raise ValueError("HKBU_API_KEY not found in environment variables")

gemini_models = [
    {
        "model": "gemini-1.5-pro",
        "api-version": "002"
    },
    {
        "model": "gemini-1.5-flash",
        "api-version": "002"
    }
]

def Gemini(
    message: str,
    model_name: str = "gemini-1.5-flash",
    temperature: float = 0.5,
    maxOutputTokens: int = 10,
    response_mime_type: str = "application/json", # "text/x.enum"
    response_schema: dict = None
):
    # Find the model in the gemini_models list
    model_info = next((model for model in gemini_models if model["model"] == model_name), None)
    
    if not model_info:
        raise ValueError(f"Model {model_name} not found in gemini_models list")
    
    api_version = model_info['api-version']
    
    url = f"{basicUrl}/deployments/{model_name}/generate_content?api-version={api_version}"
    headers = {
        'Content-Type': 'application/json',
        'accept': 'application/json',
        'api-key': apiKey
    }
    
    # Define the contents with the user's message
    contents = [{"role": "user", "parts": [{"text": message}]}]
    
    # Define the payload with the generation config and optional response schema
    payload = {
        'contents': contents,
        'generationConfig': {
            'maxOutputTokens': maxOutputTokens,
            'temperature': temperature,
            "response_mime_type": response_mime_type,
        },
        'stream': False
    }
    
    if response_schema:
        payload['generationConfig']['response_schema'] = response_schema
    
    # Make the POST request to the API
    response = requests.post(url, json=payload, headers=headers)
    
    # Check the response status code
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        print(f"Error: {response.status_code}")
        print(f"Response: {response.text}")
        return None
    