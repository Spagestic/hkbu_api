import requests
import os
from dotenv import load_dotenv
import json
from typing import Optional, Dict, List, Union, Any

load_dotenv()
apiKey = os.getenv("HKBU_API_KEY")
basicUrl = os.getenv("HKBU_BASIC_URL")
if apiKey is None:
    raise ValueError("HKBU_API_KEY not found in environment variables")
if basicUrl is None:
    raise ValueError("HKBU_BASIC_URL not found in environment variables")


qwen_models = [
    {
        "model": "qwen-max",
        'api-version': 'v1',
    },
    {
        "model": "qwen-plus",
        'api-version': 'v1',
    },
]

def get_model_info(model_name: str) -> Dict[str, str]:
    """
    Retrieve model information from the qwen_models list.

    Args:
        model_name (str): Name of the model to retrieve.

    Returns:
        Dict[str, str]: Model configuration.

    Raises:
        ValueError: If the model is not found.
    """
    model_info = next((model for model in qwen_models if model["model"] == model_name), None)
    if not model_info:
        raise ValueError(f"Model '{model_name}' not found in qwen_models list")
    return model_info

def build_conversation(message: str, system_message: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Build the conversation payload for the API request.

    Args:
        message (str): User message content.
        system_message (Optional[str]): Optional system message to set context.

    Returns:
        List[Dict[str, str]]: Conversation payload.
    """
    conversation: List[Dict[str, str]] = []
    if system_message:
        conversation.append({"role": "system", "content": system_message})
    conversation.append({"role": "user", "content": message})
    return conversation

def Qwen(
    model_name: str,
    messages: List[Dict[str, str]],
    temperature: Optional[float] = 0.7,
    max_tokens: Optional[int] = 512,
    top_p: Optional[float] = 1.0,
    frequency_penalty: Optional[float] = 0.0,
    presence_penalty: Optional[float] = 0.0,
) -> Dict[str, Any]:
    """
    Send a request to the Qwen API.

    Args:
        model (str): Model name.
        messages (List[Dict[str, str]]): Conversation messages.
        temperature (Optional[float]): Sampling temperature.
        max_tokens (Optional[int]): Maximum number of tokens in the response.
        top_p (Optional[float]): Nucleus sampling parameter.
        frequency_penalty (Optional[float]): Frequency penalty parameter.
        presence_penalty (Optional[float]): Presence penalty parameter.

    Returns:
        Dict[str, Any]: API response.
    """
    url = f"{basicUrl}/deployments/{model_name}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {apiKey}",
    }
    data = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
    }
    
    response = requests.post(url, headers=headers, json=data)
    
    if response.status_code != 200:
        raise ValueError(f"Request failed with status code {response.status_code}: {response.text}")
    
    return response.json()
