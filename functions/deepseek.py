import requests
import os
from dotenv import load_dotenv
from typing import Optional, Dict, List, Union, Any

# Load environment variables
load_dotenv()

# Validate environment variables
def validate_env_vars() -> None:
    required_vars = ["HKBU_API_KEY", "HKBU_BASIC_URL"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(f"Missing environment variables: {', '.join(missing_vars)}")

validate_env_vars()

# Constants
API_KEY: str = os.getenv("HKBU_API_KEY")  # type: ignore
BASE_URL: str = os.getenv("HKBU_BASIC_URL")  # type: ignore

# Model configurations
DEEPSEEK_MODELS: List[Dict[str, str]] = [
    {"model": "deepseek-r1", "api-version": "2024-05-01-preview"},
    {"model": "deepseek-v3", "api-version": "2024-05-01-preview"},
]

def get_model_info(model_name: str) -> Dict[str, str]:
    """
    Retrieve model information from the DEEPSEEK_MODELS list.
    
    Args:
        model_name (str): Name of the model to retrieve.
        
    Returns:
        Dict[str, str]: Model configuration.
        
    Raises:
        ValueError: If the model is not found.
    """
    model_info = next((model for model in DEEPSEEK_MODELS if model["model"] == model_name), None)
    if not model_info:
        raise ValueError(f"Model '{model_name}' not found in DEEPSEEK_MODELS list")
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


def DeepSeek(
    message: str,
    model_name: str = "deepseek-r1",
    temperature: float = 0.0,
    max_tokens: int = 255,
    stream: bool = False,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    top_p: float = 1.0,
    response_format: Optional[Dict[str, str]] = None,
    stop: Optional[List[str]] = None,
    seed: Optional[int] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    system_message: Optional[str] = None,
) -> Union[Dict[str, Any], str]:
    """
    Send a request to Azure DeepSeek service.
    
    Args:
        message (str): User message content.
        model_name (str): DeepSeek model to use.
        temperature (float): Sampling temperature (0-2.0).
        max_tokens (int): Maximum tokens in response.
        stream (bool): Whether to stream the response.
        frequency_penalty (float): Penalty for frequent tokens.
        presence_penalty (float): Penalty for token presence.
        top_p (float): Nucleus sampling threshold.
        response_format (Optional[Dict[str, str]]): Format of the response.
        stop (Optional[List[str]]): Stop sequences.
        seed (Optional[int]): Random seed for reproducibility.
        tools (Optional[List[Dict[str, Any]]]): Optional list of tools.
        system_message (Optional[str]): Optional system message to set context.
        
    Returns:
        Union[Dict[str, Any], str]: Response data from the API or error message.
    """
    try:
        # Get model info
        model_info = get_model_info(model_name)
        api_version = model_info["api-version"]

        # Build conversation
        conversation = build_conversation(message, system_message)

        # Construct URL and headers
        url = f"{BASE_URL}/deployments/{model_name}/chat/completions/?api-version={api_version}"
        headers = {"Content-Type": "application/json", "api-key": API_KEY}

        # Prepare payload
        payload: Dict[str, Any] = {
            "messages": conversation,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "top_p": top_p,
        }

        # Add optional parameters if provided
        if response_format:
            payload["response_format"] = response_format
        if stop:
            payload["stop"] = stop
        if seed is not None:
            payload["seed"] = seed
        if tools:
            payload["tools"] = tools

        # Make API request
        print(f"Sending request to {url} with payload: {payload}")
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()

        # Return parsed JSON response
        return response.json()

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return f"Error: {str(e)}"
    except ValueError as ve:
        print(f"ValueError: {ve}")
        return f"Error: {ve}"
    except Exception as ex:
        print(f"Unexpected error: {ex}")
        return f"Unexpected error: {ex}"