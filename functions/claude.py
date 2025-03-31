import os
from dotenv import load_dotenv
import requests
from typing import Optional, Dict, List, Union, Any

# Load environment variables
load_dotenv()

# Retrieve API key and base URL from environment variables
API_KEY: Optional[str] = os.getenv("HKBU_API_KEY")
BASE_URL: Optional[str] = os.getenv("HKBU_BASIC_URL")

if API_KEY is None:
    raise ValueError("HKBU_API_KEY not found in environment variables")
if BASE_URL is None:
    raise ValueError("HKBU_BASIC_URL not found in environment variables")

# Define the list of Claude models
CLAUDE_MODELS: List[Dict[str, str]] = [
    {"model": "claude-3-5-sonnet", "api-version": "20240620"},
    {"model": "claude-3-haiku", "api-version": "20240307"}
]

def find_model_info(model_name: str) -> Optional[Dict[str, str]]:
    """
    Finds the model information for a given model name.
    
    Args:
        model_name (str): The name of the model to find.
    
    Returns:
        Optional[Dict[str, str]]: The model information if found, otherwise None.
    """
    return next((model for model in CLAUDE_MODELS if model["model"] == model_name), None)

def Claude(
    message: str,
    model_name: str = "claude-3-5-sonnet",
    image_url: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 100
) -> Union[Dict[str, Any], str]:
    """
    Sends a request to the Claude API with the specified parameters.
    
    Args:
        message (str): The text message to send to the model.
        model_name (str): The name of the model to use (default: "claude-3-5-sonnet").
        image_url (Optional[str]): The URL of an image to include in the request (optional).
        temperature (float): The sampling temperature for the model (default: 0.0).
        max_tokens (int): The maximum number of tokens to generate (default: 100).
    
    Returns:
        Union[Dict[str, Any], str]: The JSON response from the API or an error message.
    """
    # Validate model name
    model_info: Optional[Dict[str, str]] = find_model_info(model_name)
    if not model_info:
        raise ValueError(f"Model {model_name} not found in CLAUDE_MODELS list")
    
    api_version: str = model_info["api-version"]
    
    # Construct the conversation payload
    conversation: List[Dict[str, Any]] = [{"role": "user", "content": message}]
    
    if image_url:
        conversation[0]["content"] = [
            {"type": "text", "text": message},
            {"type": "image_url", "image_url": {"url": image_url, "detail": "low"}}
        ]
    
    # Construct the API URL and headers
    url: str = f"{BASE_URL}/deployments/{model_name}/messages/?api-version={api_version}"
    headers: Dict[str, str] = {
        "Content-Type": "application/json",
        "api-key": API_KEY
    }
    
    # Construct the payload
    payload: Dict[str, Any] = {
        "messages": conversation,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    # Send the request
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json()
    except requests.exceptions.RequestException as e:
        return f"Error: {e}"
