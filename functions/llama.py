import requests
import os
from dotenv import load_dotenv
from typing import Optional, Dict, List, Union, Any

# Load environment variables from .env file
load_dotenv()

# Retrieve API key and base URL from environment variables
API_KEY: Optional[str] = os.getenv("HKBU_API_KEY")
BASE_URL: Optional[str] = os.getenv("HKBU_BASIC_URL")

if API_KEY is None:
    raise ValueError("HKBU_API_KEY not found in environment variables")
if BASE_URL is None:
    raise ValueError("HKBU_BASIC_URL not found in environment variables")

# Define available models
LLAMA_MODELS: List[Dict[str, str]] = [
    {
        "model": "llama3_1",
        "api-version": "20240723"
    }
]

def llama(
    message: str,
    model_name: str = "llama3_1",
    image_url: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 100,
    top_p: float = 1.0,
    top_k: int = 50,
    stop_sequences: Optional[List[str]] = None,
    stream: bool = False,
    system: Optional[str] = None
) -> Union[Dict[str, Any], str]:
    """
    Sends a request to the LLaMA model API and returns the response.

    Args:
        message (str): The input message for the model.
        model_name (str): The name of the model to use (default: "llama3_1").
        image_url (Optional[str]): URL of an image to include in the request.
        temperature (float): Sampling temperature for the model.
        max_tokens (int): Maximum number of tokens to generate.
        top_p (float): Top-p (nucleus) sampling parameter.
        top_k (int): Top-k sampling parameter.
        stop_sequences (Optional[List[str]]): List of sequences to stop generation at.
        stream (bool): Whether to stream the response.
        system (Optional[str]): System-level instructions for the model.

    Returns:
        Union[Dict[str, Any], str]: The JSON response from the API or an error message.
    """
    # Validate the model name
    model_info = next((model for model in LLAMA_MODELS if model["model"] == model_name), None)
    if not model_info:
        raise ValueError(f"Model {model_name} not found in LLAMA_MODELS list")
    
    api_version: str = model_info['api-version']

    # Construct the conversation payload
    conversation: List[Dict[str, Any]] = [{"role": "user", "content": message}]
    if image_url:
        conversation[0]["content"] = [
            {"type": "text", "text": message},
            {"type": "image_url", "image_url": {"url": image_url, "detail": "low"}}
        ]

    # Construct the API endpoint URL
    url: str = f"{BASE_URL}/deployments/{model_name}/llama/completion/?api-version={api_version}"

    # Prepare headers and payload
    headers: Dict[str, str] = {
        'Content-Type': 'application/json',
        'api-key': API_KEY
    }
    payload: Dict[str, Any] = {
        'messages': conversation,
        'temperature': temperature,
        'max_tokens': max_tokens,
        'top_p': top_p,
        'top_k': top_k,
        'stop_sequences': stop_sequences,
        'stream': stream,
        'system': system
    }

    try:
        # Send the POST request to the API
        response = requests.post(url, json=payload, headers=headers)

        # Check the response status code
        if response.status_code == 200:
            return response.json()
        else:
            return f'Error: {response.status_code}, {response.text}'
    except requests.exceptions.RequestException as e:
        return f'Network Error: {str(e)}'