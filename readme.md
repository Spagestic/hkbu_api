# HKBU GenAI Platform API Service - Example Usage

This repository provides example code snippets and usage instructions for interacting with the **HKBU GenAI Platform API Service**. The platform supports multiple AI models, including OpenAI, Claude, Gemini, and Llama, via REST API endpoints. These examples demonstrate how to query different models using their respective configurations.

## Table of Contents

- [HKBU GenAI Platform API Service - Example Usage](#hkbu-genai-platform-api-service---example-usage)
  - [Table of Contents](#table-of-contents)
  - [Prerequisites](#prerequisites)
  - [Supported Models](#supported-models)
    - [OpenAI Models](#openai-models)
    - [Claude Models](#claude-models)
    - [Gemini Models](#gemini-models)
    - [Llama Models](#llama-models)
  - [API Endpoints](#api-endpoints)
  - [Example Usage](#example-usage)
    - [OpenAI Models](#openai-models-1)
    - [Claude Models](#claude-models-1)
    - [Gemini Models](#gemini-models-1)
      - [Structured Output](#structured-output)
    - [Llama Models](#llama-models-1)
  - [Combined Query Function](#combined-query-function)
  - [Error Handling](#error-handling)
  - [Contributing](#contributing)

---

## Prerequisites

Before running the examples, ensure you have the following:

- Python 3.8 or higher installed.
- Install the required libraries:

```bash
  pip install requests python-dotenv
```

- Set up environment variables:
  - Create a `.env` file in the project root directory.
  - Add your API key and base URL:

```env
    HKBU_API_KEY=your_api_key_here
    HKBU_BASIC_URL=https://your-api-endpoint.com
```

---

## Supported Models

The HKBU GenAI Platform supports the following models:

### OpenAI Models

- `gpt-4-o`
- `gpt-4-o-mini`
- `o1-preview`
- `o1-mini`
- `text-embedding-3-large`
- `text-embedding-3-small`

### Claude Models

- `claude-3-5-sonnet`
- `claude-3-haiku`

### Gemini Models

- `gemini-1.5-pro`
- `gemini-1.5-flash`

### Llama Models

- `llama3_1`

---

## API Endpoints

Each model type has its own endpoint structure:

- **OpenAI**: `/deployments/{model_name}/chat/completions/?api-version={api_version}`
- **Claude**: `/deployments/{model_name}/messages/?api-version={api_version}`
- **Gemini**: `/deployments/{model_name}/generate_content?api-version={api_version}`
- **Llama**: `/deployments/{model_name}/llama/completion/?api-version={api_version}`

---

## Example Usage

Below are examples of how to use the API for each model type.

### OpenAI Models

```python
from dotenv import load_dotenv
import os
import requests
import json

load_dotenv()
apiKey = os.getenv("HKBU_API_KEY")
basicUrl = os.getenv("HKBU_BASIC_URL")

def query_openai_model(
    message, 
    model_name="gpt-4-o-mini", 
    temperature=0.5, 
    max_tokens=10
    ):
    url = f"{basicUrl}/deployments/{model_name}/chat/completions/?api-version=2024-10-21"
    headers = {'Content-Type': 'application/json', 'api-key': apiKey}
    payload = {
        'messages': [{"role": "user", "content": message}],
        'temperature': temperature,
        'max_tokens': max_tokens
    }
    response = requests.post(url, json=payload, headers=headers)
    return response.json()

result = query_openai_model(
    message="hello", 
    model_name="gpt-4-o-mini", 
    temperature=0.5, 
    max_tokens=10
    )
print(json.dumps(result, indent=4))
```

### Claude Models

```python
def query_claude_model(
    message, 
    model_name="claude-3-haiku", 
    temperature=0.5, 
    max_tokens=10
    ):
    url = f"{basicUrl}/deployments/{model_name}/messages/?api-version=20240307"
    headers = {'Content-Type': 'application/json', 'api-key': apiKey}
    payload = {
        'messages': [{"role": "user", "content": message}],
        'temperature': temperature,
        'max_tokens': max_tokens
    }
    response = requests.post(url, json=payload, headers=headers)
    return response.json()

result = query_claude_model(
    message="hello", 
    model_name="claude-3-haiku", 
    temperature=0.5, 
    max_tokens=10
    )
print(json.dumps(result, indent=4))
```

### Gemini Models

```python
def query_gemini_model(
    message, 
    model_name="gemini-1.5-flash", 
    temperature=0.5, 
    maxOutputTokens=10
    ):
    url = f"{basicUrl}/deployments/{model_name}/generate_content?api-version=002"
    headers = {'Content-Type': 'application/json', 'accept': 'application/json', 'api-key': apiKey}
    payload = {
        'contents': [{"role": "user", "parts": [{"text": message}]}],
        'generationConfig': {
            'maxOutputTokens': maxOutputTokens,
            'temperature': temperature
        }
    }
    response = requests.post(url, json=payload, headers=headers)
    return response.json()

result = query_gemini_model(
    message="hello", 
    model_name="gemini-1.5-flash", 
    temperature=0.5, 
    maxOutputTokens=10
    )
print(json.dumps(result, indent=4))
```

#### Structured Output

The Gemini model also supports structured outputs. You can specify a `response_schema` parameter to define the structure of the output.

```python
def query_gemini_model_with_schema(
    message, 
    model_name="gemini-1.5-flash", 
    temperature=0.5, 
    maxOutputTokens=10, 
    response_schema=None
    ):
    url = f"{basicUrl}/deployments/{model_name}/generate_content?api-version=002"
    headers = {'Content-Type': 'application/json', 'accept': 'application/json', 'api-key': apiKey}
    payload = {
        'contents': [{"role": "user", "parts": [{"text": message}]}],
        'generationConfig': {
            'maxOutputTokens': maxOutputTokens,
            'temperature': temperature
        }
    }
    if response_schema:
        payload['response_schema'] = response_schema
    response = requests.post(url, json=payload, headers=headers)
    return response.json()

schema = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "keywords": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["summary", "keywords"]
}

result = query_gemini_model_with_schema(
    message="hello", 
    model_name="gemini-1.5-flash", 
    temperature=0.5, 
    maxOutputTokens=10, 
    response_schema=schema
    )
print(json.dumps(result, indent=4))
```

### Llama Models

```python
def query_llama_model(
    message, 
    model_name="llama3_1", 
    temperature=0.5, 
    max_tokens=10
    ):
    url = f"{basicUrl}/deployments/{model_name}/llama/completion/?api-version=20240723"
    headers = {'Content-Type': 'application/json', 'api-key': apiKey}
    payload = {
        'messages': [{"role": "user", "content": message}],
        'temperature': temperature,
        'max_tokens': max_tokens
    }
    response = requests.post(url, json=payload, headers=headers)
    return response.json()

result = query_llama_model(
    message="hello", 
    model_name="llama3_1", 
    temperature=0.5, 
    max_tokens=10
    )
print(json.dumps(result, indent=4))
```

---

## Combined Query Function

A unified function `query_model` is provided to simplify querying across all model types. Here's an example:

```python
result = query_model(
    message="hello",
    model_name="gemini-1.5-flash",
    kwargs={
        'temperature': 0.5,
        'maxOutputTokens': 6
    }
)
print(json.dumps(result, indent=4))
```

---

## Error Handling

If the API request fails, the response will include an error message. For example:

```python
if isinstance(result, tuple) and result[0] == 'Error:':
    print(f"Error: {result[1].status_code} - {result[1].text}")
else:
    print(json.dumps(result, indent=4))
```

---

## Contributing

We welcome contributions! If you find any issues or have suggestions for improvement, please open an issue or submit a pull request.
