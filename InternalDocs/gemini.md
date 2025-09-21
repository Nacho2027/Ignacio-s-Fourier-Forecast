# Complete Guide to Using Gemini 2.5 Pro with Google API for Python

## Table of Contents
1. [Prerequisites and Setup](#prerequisites-and-setup)
2. [Installation](#installation)
3. [Authentication and Client Setup](#authentication-and-client-setup)
4. [Basic Text Generation](#basic-text-generation)
5. [Configuration Parameters](#configuration-parameters)
6. [Function Calling](#function-calling)
7. [Multimodal Input (Images and Files)](#multimodal-input-images-and-files)
8. [Streaming Responses](#streaming-responses)
9. [Safety Settings](#safety-settings)
10. [Best Practices and Prompting Tips](#best-practices-and-prompting-tips)
11. [Rate Limits and Error Handling](#rate-limits-and-error-handling)
12. [Advanced Features](#advanced-features)

## Prerequisites and Setup

Before getting started, you'll need:
- Python 3.9 or higher
- A Gemini API key (get it free from [Google AI Studio](https://aistudio.google.com/app/apikey))

## Installation

Install the Google GenAI SDK using pip:

```bash
pip install -q -U google-genai
```

## Authentication and Client Setup

### Method 1: Environment Variable (Recommended)
Set your API key as an environment variable:

```bash
export GEMINI_API_KEY='your-api-key-here'
```

Then create a client:
```python
from google import genai

# Client automatically picks up GEMINI_API_KEY from environment
client = genai.Client()
```

### Method 2: Direct API Key
```python
from google import genai

client = genai.Client(api_key='your-api-key-here')
```

### Method 3: Vertex AI (Enterprise)
For Google Cloud Vertex AI:
```python
from google import genai

client = genai.Client(
    vertexai=True, 
    project='your-project-id', 
    location='us-central1'
)
```

## Basic Text Generation

### Simple Text Generation
```python
from google import genai

client = genai.Client()

response = client.models.generate_content(
    model="gemini-2.5-pro",
    contents="Explain quantum computing in simple terms"
)

print(response.text)
```

### With System Instructions
```python
from google.genai import types

response = client.models.generate_content(
    model="gemini-2.5-pro",
    contents="What's the weather like?",
    config=types.GenerateContentConfig(
        system_instruction="You are a helpful weather assistant. Always ask for the user's location if not provided."
    )
)

print(response.text)
```

## Configuration Parameters

### Essential Parameters
```python
from google.genai import types

config = types.GenerateContentConfig(
    # Temperature: Controls randomness (0.0-2.0)
    # 0.0 = deterministic, 1.0 = default, 2.0 = very creative
    temperature=0.7,
    
    # Top-P: Controls diversity (0.0-1.0)
    # Lower values = more focused, higher = more diverse
    top_p=0.8,
    
    # Top-K: Limits token selection to top K most probable
    top_k=40,
    
    # Maximum output tokens
    max_output_tokens=1000,
    
    # Stop sequences to end generation
    stop_sequences=["END", "STOP"],
    
    # Seed for reproducible results
    seed=42
)

response = client.models.generate_content(
    model="gemini-2.5-pro",
    contents="Write a short story",
    config=config
)
```

### Thinking Configuration
Gemini 2.5 Pro has "thinking" enabled by default. Control it with:

```python
config = types.GenerateContentConfig(
    # Disable thinking for faster responses
    thinking_config=types.ThinkingConfig(thinking_budget=0)
)
```

## Function Calling

### Automatic Function Calling (Recommended)
```python
from google import genai
from google.genai import types

def get_weather(location: str) -> str:
    """Gets the current weather for a location.
    
    Args:
        location: The city and state, e.g. San Francisco, CA
    
    Returns:
        A description of the weather
    """
    # Your weather API logic here
    return f"Sunny, 72°F in {location}"

def calculate_tip(bill_amount: float, tip_percentage: float = 15.0) -> dict:
    """Calculates tip amount and total bill.
    
    Args:
        bill_amount: The bill amount in dollars
        tip_percentage: Tip percentage (default 15%)
    
    Returns:
        Dictionary with tip amount and total
    """
    tip = bill_amount * (tip_percentage / 100)
    total = bill_amount + tip
    return {"tip_amount": tip, "total_amount": total, "tip_percentage": tip_percentage}

# Configure with automatic function calling
client = genai.Client()
config = types.GenerateContentConfig(
    tools=[get_weather, calculate_tip]  # Pass functions directly
)

response = client.models.generate_content(
    model="gemini-2.5-pro",
    contents="What's the weather in Boston and calculate a 20% tip on a $50 bill?",
    config=config
)

print(response.text)
```

### Manual Function Calling
```python
from google.genai import types

# Define function schema manually
weather_function = {
    "name": "get_weather",
    "description": "Gets the current weather for a location",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA"
            }
        },
        "required": ["location"]
    }
}

tools = types.Tool(function_declarations=[weather_function])
config = types.GenerateContentConfig(tools=[tools])

response = client.models.generate_content(
    model="gemini-2.5-pro",
    contents="What's the weather in New York?",
    config=config
)

# Check for function call
if response.candidates[0].content.parts[0].function_call:
    function_call = response.candidates[0].content.parts[0].function_call
    print(f"Function: {function_call.name}")
    print(f"Args: {function_call.args}")
    
    # Execute your function here and send result back
    # ... (function execution logic)
```

### Function Calling Modes
```python
# Force function calling
tool_config = types.ToolConfig(
    function_calling_config=types.FunctionCallingConfig(
        mode="ANY",  # Always call a function
        allowed_function_names=["get_weather"]  # Restrict to specific functions
    )
)

config = types.GenerateContentConfig(
    tools=[get_weather],
    tool_config=tool_config
)
```

## Multimodal Input (Images and Files)

### Image Processing
```python
from google.genai import types
from PIL import Image

# From local file
def analyze_local_image():
    with open('image.jpg', 'rb') as f:
        image_bytes = f.read()
    
    image_part = types.Part.from_bytes(
        data=image_bytes, 
        mime_type='image/jpeg'
    )
    
    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=["Describe what you see in this image", image_part]
    )
    
    return response.text

# From URL
import requests

def analyze_image_from_url(image_url: str):
    image_bytes = requests.get(image_url).content
    
    image = types.Part.from_bytes(
        data=image_bytes, 
        mime_type="image/jpeg"
    )
    
    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=["What objects can you identify in this image?", image]
    )
    
    return response.text
```

### File Upload and Processing (Gemini Developer API only)
```python
# Upload a file
file = client.files.upload(file='document.pdf')

response = client.models.generate_content(
    model="gemini-2.5-pro",
    contents=['Summarize this document', file]
)

print(response.text)

# Clean up
client.files.delete(name=file.name)
```

### Object Detection (Gemini 2.0+)
```python
from google.genai import types
from PIL import Image
import json

# Enable object detection with bounding boxes
prompt = "Detect all prominent items in the image. Provide bounding boxes as [ymin, xmin, ymax, xmax] normalized to 0-1000."

image = Image.open("image.jpg")

config = types.GenerateContentConfig(
    response_mime_type="application/json"
)

response = client.models.generate_content(
    model="gemini-2.5-pro",
    contents=[image, prompt],
    config=config
)

# Convert normalized coordinates to absolute coordinates
width, height = image.size
bounding_boxes = json.loads(response.text)

for bbox in bounding_boxes:
    abs_y1 = int(bbox["box_2d"][0]/1000 * height)
    abs_x1 = int(bbox["box_2d"][1]/1000 * width)
    abs_y2 = int(bbox["box_2d"][2]/1000 * height)
    abs_x2 = int(bbox["box_2d"][3]/1000 * width)
    print(f"Object: {bbox.get('name', 'Unknown')}, Box: [{abs_x1}, {abs_y1}, {abs_x2}, {abs_y2}]")
```

## Streaming Responses

### Synchronous Streaming
```python
print("Response: ", end="")
for chunk in client.models.generate_content_stream(
    model="gemini-2.5-pro",
    contents="Tell me a long story about space exploration"
):
    print(chunk.text, end="", flush=True)
print()  # New line at the end
```

### Asynchronous Streaming
```python
import asyncio

async def stream_response():
    print("Response: ", end="")
    async for chunk in await client.aio.models.generate_content_stream(
        model="gemini-2.5-pro",
        contents="Explain the history of artificial intelligence"
    ):
        print(chunk.text, end="", flush=True)
    print()

# Run the async function
asyncio.run(stream_response())
```

## Safety Settings

```python
from google.genai import types

# Configure safety settings
safety_settings = [
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH
    )
]

config = types.GenerateContentConfig(safety_settings=safety_settings)

response = client.models.generate_content(
    model="gemini-2.5-pro",
    contents="Your prompt here",
    config=config
)
```

Available thresholds:
- `BLOCK_NONE`: No blocking
- `BLOCK_ONLY_HIGH`: Block only high-risk content
- `BLOCK_MEDIUM_AND_ABOVE`: Block medium and high-risk content
- `BLOCK_LOW_AND_ABOVE`: Block low, medium, and high-risk content

## Best Practices and Prompting Tips

### 1. Use the PTCF Framework
Structure prompts using **Persona, Task, Context, Format**:

```python
prompt = """
Persona: You are an expert data scientist with 10 years of experience.

Task: Analyze the following dataset and identify the top 3 insights.

Context: This is sales data from an e-commerce company for Q4 2024. The company wants to improve their marketing strategy.

Format: Present your findings as a numbered list with brief explanations for each insight.

Data: [your data here]
"""
```

### 2. Optimize Temperature Settings
- **Creative tasks**: 0.8-1.0 temperature
- **Analytical tasks**: 0.2-0.5 temperature  
- **Code generation**: 0.0-0.3 temperature
- **Conversational**: 0.4-0.7 temperature

### 3. Effective System Instructions
```python
system_instruction = """
You are a professional code reviewer. When reviewing code:
1. Focus on readability, performance, and security
2. Provide specific suggestions for improvement
3. Explain the reasoning behind your recommendations
4. Use a constructive and educational tone
"""
```

### 4. Handle Long Context Effectively
Gemini 2.5 Pro supports up to 1M tokens:

```python
# For large documents, use structured prompts
prompt = """
Please analyze this document and provide:
1. Executive summary (2-3 sentences)
2. Key findings (bullet points)
3. Recommendations (numbered list)

Document: {your_large_document}
"""
```

### 5. Iterative Refinement
```python
def refine_response(initial_prompt: str, feedback: str) -> str:
    refined_prompt = f"""
    Original request: {initial_prompt}
    
    Previous response feedback: {feedback}
    
    Please provide an improved response that addresses the feedback.
    """
    
    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=refined_prompt,
        config=types.GenerateContentConfig(temperature=0.3)
    )
    
    return response.text
```

## Rate Limits and Error Handling

### Understanding Rate Limits
- **Free Tier**: Limited requests per minute/day
- **Tier 1** (with billing): Higher limits
- **Tier 2** ($250+ spent): Even higher limits
- **Tier 3** ($1000+ spent): Highest limits

### Error Handling
```python
from google.genai import errors
import time
import random

def generate_with_retry(prompt: str, max_retries: int = 3):
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-pro",
                contents=prompt
            )
            return response.text
            
        except errors.APIError as e:
            if e.code == 429:  # Rate limit exceeded
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                print(f"Rate limited. Waiting {wait_time:.2f} seconds...")
                time.sleep(wait_time)
            elif e.code == 500:  # Server error
                print(f"Server error: {e.message}")
                time.sleep(1)
            else:
                print(f"API Error {e.code}: {e.message}")
                break
                
        except Exception as e:
            print(f"Unexpected error: {e}")
            break
    
    return None

# Usage
result = generate_with_retry("Explain machine learning")
```

### Token Counting
```python
# Count tokens before sending
token_count = client.models.count_tokens(
    model="gemini-2.5-pro",
    contents="Your prompt here"
)

print(f"Token count: {token_count.total_tokens}")
```

## Advanced Features

### Chat Sessions
```python
# Create a persistent chat session
chat = client.chats.create(
    model="gemini-2.5-pro",
    config=types.GenerateContentConfig(
        system_instruction="You are a helpful coding assistant."
    )
)

# Multi-turn conversation
response1 = chat.send_message("How do I create a REST API in Python?")
print("Assistant:", response1.text)

response2 = chat.send_message("Can you show me an example using FastAPI?")
print("Assistant:", response2.text)

response3 = chat.send_message("How do I add authentication to it?")
print("Assistant:", response3.text)
```

### JSON Response Schema
```python
from pydantic import BaseModel
from google.genai import types

class ProductInfo(BaseModel):
    name: str
    price: float
    category: str
    in_stock: bool
    description: str

config = types.GenerateContentConfig(
    response_mime_type="application/json",
    response_schema=ProductInfo
)

response = client.models.generate_content(
    model="gemini-2.5-pro",
    contents="Generate information for a laptop product",
    config=config
)

# Response will be valid JSON matching the schema
import json
product = json.loads(response.text)
print(product)
```

### Caching for Large Context
```python
from google.genai import types

# Create cached content for large documents
cached_content = client.caches.create(
    model="gemini-2.5-pro",
    config=types.CreateCachedContentConfig(
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text("Large document content here...")
                ]
            )
        ],
        system_instruction="Analyze this document for key insights",
        display_name="document_analysis_cache",
        ttl="3600s"  # Cache for 1 hour
    )
)

# Use cached content in subsequent requests
response = client.models.generate_content(
    model="gemini-2.5-pro",
    contents="What are the main themes in this document?",
    config=types.GenerateContentConfig(
        cached_content=cached_content.name
    )
)
```

### Batch Processing
```python
# For processing multiple requests efficiently
batch_job = client.batches.create(
    model="gemini-2.5-pro",
    src=[
        {
            "contents": [{"parts": [{"text": "What is AI?"}], "role": "user"}],
            "config": {"response_modalities": ["text"]}
        },
        {
            "contents": [{"parts": [{"text": "Explain quantum physics"}], "role": "user"}],
            "config": {"response_modalities": ["text"]}
        }
    ]
)

# Monitor job status
import time
while batch_job.state not in ['JOB_STATE_SUCCEEDED', 'JOB_STATE_FAILED']:
    time.sleep(10)
    batch_job = client.batches.get(name=batch_job.name)
    print(f"Job state: {batch_job.state}")
```

## Complete Example: AI-Powered Code Reviewer

```python
from google import genai
from google.genai import types
import json

class CodeReviewer:
    def __init__(self, api_key: str = None):
        self.client = genai.Client(api_key=api_key)
        
    def review_code(self, code: str, language: str = "python") -> dict:
        """Review code and return structured feedback."""
        
        system_instruction = f"""
        You are an expert {language} code reviewer. Analyze the provided code and return feedback in the specified JSON format.
        Focus on:
        1. Code quality and readability
        2. Performance optimizations
        3. Security vulnerabilities
        4. Best practices adherence
        5. Potential bugs
        """
        
        class CodeReview(BaseModel):
            overall_score: int  # 1-10
            issues: list[dict]  # [{"type": "bug|style|performance|security", "line": int, "description": str, "suggestion": str}]
            strengths: list[str]
            summary: str
        
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=CodeReview,
            temperature=0.1  # Low temperature for consistent analysis
        )
        
        prompt = f"""
        Please review this {language} code:
        
        ```{language}
        {code}
        ```
        """
        
        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-pro",
                contents=prompt,
                config=config
            )
            
            return json.loads(response.text)
            
        except Exception as e:
            return {"error": f"Review failed: {str(e)}"}

# Usage example
reviewer = CodeReviewer()

code_to_review = """
def calculate_average(numbers):
    sum = 0
    for i in range(len(numbers)):
        sum += numbers[i]
    return sum / len(numbers)

def process_data(data):
    result = []
    for item in data:
        if item > 0:
            result.append(calculate_average(item))
    return result
"""

review_result = reviewer.review_code(code_to_review)
print(json.dumps(review_result, indent=2))
```

This guide covers all essential aspects of using Gemini 2.5 Pro with the Google API for Python. The examples are based on official Google documentation and best practices from the community. Remember to always handle API keys securely and implement proper error handling in production applications.