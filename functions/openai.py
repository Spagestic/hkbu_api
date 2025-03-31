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

openai_models = [
    {
        "model": "gpt-4-o",
        'api-version': '2024-10-21',
    },
    {
        "model": "gpt-4-o-mini",
        'api-version': '2024-10-21',
    },
    {
        "model": "o1-preview",
        'api-version': '2024-10-21',
    },
    {
        "model": "o1-mini",
        'api-version': '2024-10-21',
    },
    {
        "model": "text-embedding-3-large",
        'api-version': '2024-05-01-preview',
    },
    {
        "model": "text-embedding-3-small",
        'api-version': '2024-05-01-preview',
    },
]

def OpenAI(
        message: str,
        model_name: str = "gpt-4-o-mini",
        imageURL: Optional[str] = None,
        temperature: float = 0,
        max_tokens: int = 100,
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        response_format: Optional[Dict[str, str]] = None,
        system_message: Optional[str] = None
        ) -> Union[Dict[str, Any], str]:
    """
    Send a request to Azure OpenAI service.
    
    Args:
        message: User message content
        model_name: OpenAI model to use
        imageURL: Optional URL to image for vision models
        temperature: Sampling temperature (0-2.0)
        max_tokens: Maximum tokens in response
        tools: List of tools for function calling
        stream: Whether to stream the response
        response_format: Specify response format (e.g., {"type": "json_object"})
        system_message: Optional system message to set context
        
    Returns:
        Response data from the API or error message
    """
    # Find the model in the openai_models list
    model_info = next((model for model in openai_models if model["model"] == model_name), None)
    
    if not model_info:
        raise ValueError(f"Model {model_name} not found in openai_models list")
    
    api_version = model_info['api-version']
    
    # Build conversation
    conversation = []
    
    if system_message:
        conversation.append({"role": "system", "content": system_message})
    
    # Add user message
    user_message = {"role": "user"}
    if imageURL:
        user_message["content"] = [
            {"type": "text", "text": message},
            {"type": "image_url", "image_url": {"url": imageURL, "detail": "low"}}
        ]
    else:
        user_message["content"] = message
    
    conversation.append(user_message)
    
    url = f"{basicUrl}/deployments/{model_name}/chat/completions/?api-version={api_version}"
    headers = {'Content-Type': 'application/json', 'api-key': apiKey}
    payload = { 
        'messages': conversation,
        'temperature': temperature,
        'max_tokens': max_tokens,
        'stream': stream,
    }
    
    # Only add optional parameters if they're provided
    if tools:
        payload['tools'] = tools
    
    if response_format:
        payload['response_format'] = response_format
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()  # Raise exception for 4XX/5XX responses
        return response.json()
    except requests.exceptions.RequestException as e:
        return f'Error: {str(e)}'