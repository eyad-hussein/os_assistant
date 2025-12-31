# Vision Module

Multimodal vision support for OS Assistant using Ollama vision models.

## Features

- **Screenshot Analysis**: Analyze error dialogs, terminal outputs, and system screenshots
- **Error Code Detection**: Automatically extract error codes from images
- **Text Extraction**: Read text directly from images (no OCR needed!)
- **Context Enhancement**: Enhance user prompts with visual information
- **Multiple Formats**: Support for PNG, JPG, JPEG, GIF, WebP, BMP

## Supported Models

| Model | Description | VRAM Required |
|-------|-------------|---------------|
| `llava:7b` | LLaVA 1.6 (recommended) | ~8GB |
| `llama3.2-vision:11b` | Llama 3.2 Vision | ~12GB |
| `bakllava:latest` | BakLLaVA variant | ~8GB |

## Configuration

Set these environment variables in `.env`:

```bash
# Enable/disable vision
VISION_ENABLED=true

# Vision model to use
VISION_MODEL=llava:7b

# Max image dimension (resized if larger)
VISION_MAX_IMAGE_SIZE=1024
```

## Usage

### Via Streamlit UI

1. Start the app: `streamlit run streamlit_app/app.py`
2. Upload a screenshot in the sidebar
3. Type your question
4. Click "Run" - the image will be analyzed automatically

### Programmatically

```python
from PIL import Image
from os_assistant.tools.vision import VisionAnalyzer

# Initialize analyzer
analyzer = VisionAnalyzer()

# Analyze an image
result = analyzer.analyze("error_screenshot.png", "What is this error?")

if result.success:
    print(f"Screenshot type: {result.screenshot_type}")
    print(f"Error codes: {result.error_codes}")
    print(f"Analysis: {result.analysis}")
    print(f"Suggestions: {result.suggested_actions}")
```

### With OSAssistant

```python
from os_assistant.os_assistant import OSAssistant

assistant = OSAssistant()

# Read image as bytes
with open("screenshot.png", "rb") as f:
    image_bytes = f.read()

# Process with image attachment
assistant.process_prompt(
    "What's wrong in this screenshot?",
    initial_state={"attached_image": image_bytes}
)
```

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    User Query + Image                        │
└─────────────────────────┬────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────┐
│                 Vision Analysis Node                         │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  1. Prepare image (resize, convert to base64)          │ │
│  │  2. Send to Ollama vision model                        │ │
│  │  3. Parse structured response                          │ │
│  │  4. Build enhanced prompt                              │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────┬────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────┐
│              Enhanced Prompt + Vision Context                │
│  - Original question                                         │
│  - Extracted text from image                                 │
│  - Detected error codes                                      │
│  - Screenshot type classification                            │
│  - Preliminary suggestions                                   │
└─────────────────────────┬────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────┐
│                  Rest of LangGraph Workflow                  │
│        (Domain Analysis → Context → Response)                │
└──────────────────────────────────────────────────────────────┘
```

## Testing

```bash
# Basic functionality tests (no Ollama required)
python scripts/test_vision_basic.py

# Full integration with Ollama
python scripts/test_vision_ollama.py

# Workflow integration tests
python scripts/test_vision_workflow.py
```

## Why No OCR?

Modern vision LLMs like LLaVA and llama3.2-vision can:
- **Read text directly** from images with context understanding
- **Identify UI elements** and their relationships
- **Provide interpretation**, not just extraction

Traditional OCR (like Tesseract) would:
- Only extract raw text without context
- Miss the semantic meaning of the screenshot
- Add extra dependencies and complexity

## Files

- `analyzer.py` - VisionAnalyzer class with image analysis logic
- `config.py` - VisionConfig dataclass for configuration
- `__init__.py` - Package exports
- `../core/nodes/vision_analysis.py` - LangGraph node integration
