import base64
import importlib.util
import threading
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Union

from langchain.schema import HumanMessage
from langchain_ollama import ChatOllama
from os_assistant.utils.settings import MODEL_TYPE

from os_assistant.utils import LOGGER

from .config import VISION_CONFIG, VisionConfig

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage


@dataclass
class VisionAnalysisResult:
    """
    Result of vision analysis on an image.

    Attributes:
        success: Whether analysis completed successfully
        extracted_text: Text detected in the image (error messages, etc.)
        error_codes: List of error codes found (e.g., 0x80070005, EACCES)
        screenshot_type: Type of screenshot (terminal, dialog, browser, etc.)
        analysis: Detailed analysis of the image content
        suggested_actions: Recommended actions based on analysis
        enhanced_prompt: Original prompt enhanced with vision insights
        raw_response: Raw LLM response for debugging
        error: Error message if analysis failed
    """

    success: bool = False
    extracted_text: str = ""
    error_codes: list[str] = field(default_factory=list)
    screenshot_type: str = "unknown"
    analysis: str = ""
    suggested_actions: list[str] = field(default_factory=list)
    enhanced_prompt: str = ""
    raw_response: str = ""
    error: str | None = None


# Analysis prompt optimized for OS troubleshooting
VISION_ANALYSIS_PROMPT = """You are an expert OS troubleshooting assistant analyzing a screenshot.

Analyze this image carefully and provide structured information:

1. **EXTRACTED_TEXT**: Any visible text (error messages, file names, paths, commands, etc.)
2. **ERROR_CODES**: List any error codes you find (e.g., 0x80070005, EACCES, errno values, HTTP status codes)
3. **SCREENSHOT_TYPE**: What type of screenshot is this?
   - terminal: Command line / shell output
   - error_dialog: System error dialog or popup
   - settings: System settings or configuration panel
   - file_manager: File browser / explorer
   - browser: Web browser window
   - ide: Code editor / IDE
   - installer: Installation wizard or package manager
   - task_manager: Process / resource monitor
   - other: Anything else

4. **ANALYSIS**: What is happening in this screenshot? What problem or situation does it show?

5. **SUGGESTED_ACTIONS**: What specific actions could address this? List 2-4 actionable suggestions.

{user_context}

RESPOND IN THIS EXACT FORMAT (no markdown, just plain text with these headers):

EXTRACTED_TEXT: [all visible text, preserve formatting if important]
ERROR_CODES: [comma-separated codes, or "none" if no codes found]
SCREENSHOT_TYPE: [one of the types listed above]
ANALYSIS: [your analysis of what's happening]
SUGGESTED_ACTIONS:
- [action 1]
- [action 2]
- [action 3]
"""


class VisionAnalyzer:
    """
    Analyzes screenshots and images using Ollama vision models.

    Supports various image formats and provides structured analysis
    optimized for OS troubleshooting and assistance.

    Example usage:
        analyzer = VisionAnalyzer()
        result = analyzer.analyze("screenshot.png", "What's wrong with my system?")
        print(result.analysis)
        print(result.suggested_actions)
    """

    def __init__(self, config: VisionConfig | None = None):
        """
        Initialize the vision analyzer.

        Args:
            config: Vision configuration, uses VISION_CONFIG if not provided
        """
        self.config = config or VISION_CONFIG
        self._model: ChatOllama | None = None
        self._pil_available: bool | None = None

        # Validate configuration at initialization
        if self.config.enabled:
            errors = self.config.validate()
            if errors:
                LOGGER.warning(f"Vision config validation issues: {', '.join(errors)}")
                self.config.enabled = False

    @property
    def model(self) -> ChatOllama:
        """Lazy-load the vision model on first use."""
        if self._model is None:
            LOGGER.info(f"Initializing vision model: {self.config.model_name}")
            # Only initialize Ollama-backed model when configured as such
            if MODEL_TYPE and MODEL_TYPE.upper() == "OLLAMA":
                self._model = ChatOllama(
                    model=self.config.model_name,
                    base_url=self.config.base_url,
                    temperature=0,  # Deterministic for consistent analysis
                )
            else:
                raise RuntimeError(
                    "Vision analysis requires MODEL_TYPE=OLLAMA and a valid MODEL_BASE_URL"
                )
        return self._model

    def _check_pil_available(self) -> bool:
        """Check if PIL/Pillow is available for image processing."""
        if self._pil_available is None:
            self._pil_available = importlib.util.find_spec("PIL.Image") is not None
            if not self._pil_available:
                LOGGER.warning(
                    "PIL/Pillow not installed. Install with: pip install Pillow"
                )
        return self._pil_available

    def is_enabled(self) -> bool:
        """Check if vision analysis is enabled and available."""
        return self.config.enabled and self._check_pil_available()

    def analyze(
        self,
        image: Union[str, Path, bytes, "PILImage"],
        user_prompt: str = "",
    ) -> VisionAnalysisResult:
        """
        Analyze an image and extract OS-relevant information.

        Args:
            image: Image as file path, bytes, base64 string, or PIL Image
            user_prompt: Optional user context to include in analysis

        Returns:
            VisionAnalysisResult with structured analysis

        Examples:
            # From file path
            result = analyzer.analyze("error_screenshot.png", "What's this error?")

            # From bytes (e.g., from file upload)
            result = analyzer.analyze(image_bytes, "Help me fix this")

            # From PIL Image
            from PIL import Image
            img = Image.open("screenshot.png")
            result = analyzer.analyze(img)
        """
        if not self.config.enabled:
            return VisionAnalysisResult(
                success=False, error="Vision analysis is disabled in configuration"
            )

        if not self._check_pil_available():
            return VisionAnalysisResult(
                success=False,
                error="PIL/Pillow not installed. Run: pip install Pillow",
            )

        try:
            # Convert image to base64
            image_b64 = self._prepare_image(image)

            # Build prompt with user context
            user_context = ""
            if user_prompt:
                user_context = f"\nUser's question/context: {user_prompt}"

            prompt = VISION_ANALYSIS_PROMPT.format(user_context=user_context)

            # Create multimodal message
            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                ]
            )

            # Invoke vision model
            LOGGER.info(f"Analyzing image with {self.config.model_name}...")
            response = self.model.invoke([message])
            raw_response = response.content

            # Parse structured response
            result = self._parse_response(raw_response, user_prompt)
            result.raw_response = raw_response
            result.success = True

            LOGGER.info(
                f"Vision analysis complete. Type: {result.screenshot_type}, "
                f"Errors found: {len(result.error_codes)}"
            )
            return result

        except Exception as e:
            LOGGER.error(f"Vision analysis failed: {e}")
            return VisionAnalysisResult(success=False, error=str(e))

    def _prepare_image(self, image: Union[str, Path, bytes, "PILImage"]) -> str:
        """
        Convert any image input to base64 string.

        Args:
            image: Image in various formats

        Returns:
            Base64-encoded JPEG string

        Raises:
            ValueError: If image type is not supported
        """
        from PIL import Image as PILImage

        # Handle string input
        if isinstance(image, str):
            # Check if already base64 (long string without path separators)
            if len(image) > 500 and "/" not in image and "\\" not in image:
                return image
            # Treat as file path
            image = Path(image)

        # Handle file path
        if isinstance(image, Path):
            if not image.exists():
                raise ValueError(f"Image file not found: {image}")
            with open(image, "rb") as f:
                image = f.read()

        # Handle bytes
        if isinstance(image, bytes):
            image = PILImage.open(BytesIO(image))

        # Handle PIL Image
        if hasattr(image, "size") and hasattr(image, "mode"):  # Duck typing for PIL
            # Resize if necessary
            if max(image.size) > self.config.max_image_size:
                ratio = self.config.max_image_size / max(image.size)
                new_size = (int(image.width * ratio), int(image.height * ratio))
                image = image.resize(new_size, PILImage.Resampling.LANCZOS)
                LOGGER.debug(f"Resized image to {new_size}")

            # Convert to RGB if necessary (handles RGBA, P, L modes)
            if image.mode not in ("RGB",):
                image = image.convert("RGB")

            # Convert to base64 JPEG
            buffer = BytesIO()
            image.save(buffer, format="JPEG", quality=self.config.jpeg_quality)
            return base64.b64encode(buffer.getvalue()).decode("utf-8")

        raise ValueError(f"Unsupported image type: {type(image)}")

    def _parse_response(
        self, response: str, original_prompt: str
    ) -> VisionAnalysisResult:
        """
        Parse LLM response into structured VisionAnalysisResult.

        Args:
            response: Raw text response from vision model
            original_prompt: Original user prompt for enhanced prompt building

        Returns:
            Parsed VisionAnalysisResult
        """
        result = VisionAnalysisResult()

        lines = response.split("\n")
        current_section = None
        suggested_actions = []
        analysis_lines = []

        for line in lines:
            line_stripped = line.strip()

            if line_stripped.startswith("EXTRACTED_TEXT:"):
                result.extracted_text = line_stripped.replace(
                    "EXTRACTED_TEXT:", ""
                ).strip()
                current_section = "text"
            elif line_stripped.startswith("ERROR_CODES:"):
                codes_str = line_stripped.replace("ERROR_CODES:", "").strip()
                if codes_str.lower() not in ("none", "n/a", ""):
                    result.error_codes = [
                        c.strip() for c in codes_str.split(",") if c.strip()
                    ]
                current_section = "codes"
            elif line_stripped.startswith("SCREENSHOT_TYPE:"):
                result.screenshot_type = (
                    line_stripped.replace("SCREENSHOT_TYPE:", "").strip().lower()
                )
                current_section = "type"
            elif line_stripped.startswith("ANALYSIS:"):
                analysis_text = line_stripped.replace("ANALYSIS:", "").strip()
                if analysis_text:
                    analysis_lines.append(analysis_text)
                current_section = "analysis"
            elif line_stripped.startswith("SUGGESTED_ACTIONS:"):
                current_section = "actions"
            elif current_section == "actions" and line_stripped.startswith("-"):
                action = line_stripped[1:].strip()
                if action:
                    suggested_actions.append(action)
            elif current_section == "analysis" and line_stripped:
                # Continue collecting analysis text
                if not line_stripped.startswith("SUGGESTED_ACTIONS"):
                    analysis_lines.append(line_stripped)

        result.analysis = " ".join(analysis_lines)
        result.suggested_actions = suggested_actions

        # Build enhanced prompt
        result.enhanced_prompt = self._build_enhanced_prompt(result, original_prompt)

        return result

    def _build_enhanced_prompt(
        self, result: VisionAnalysisResult, original_prompt: str
    ) -> str:
        """
        Build an enhanced prompt combining user query with vision analysis.

        This enhanced prompt provides the LLM with rich context from the
        screenshot analysis, enabling more accurate and relevant responses.

        Args:
            result: Vision analysis result
            original_prompt: Original user prompt

        Returns:
            Enhanced prompt string
        """
        parts = []

        # Original user prompt
        if original_prompt:
            parts.append(f"User's question: {original_prompt}")

        parts.append("\n" + "=" * 50)
        parts.append("SCREENSHOT ANALYSIS (from attached image)")
        parts.append("=" * 50)

        # Screenshot type
        parts.append(f"Screenshot type: {result.screenshot_type}")

        # Extracted text
        if result.extracted_text:
            parts.append(f"\nText found in image:\n{result.extracted_text}")

        # Error codes (highlighted)
        if result.error_codes:
            codes_str = ", ".join(result.error_codes)
            parts.append(f"\nError codes detected: {codes_str}")

        # Analysis
        if result.analysis:
            parts.append(f"\nImage analysis:\n{result.analysis}")

        # Suggested actions from vision
        if result.suggested_actions:
            parts.append("\nPreliminary suggestions from image analysis:")
            for i, action in enumerate(result.suggested_actions, 1):
                parts.append(f"  {i}. {action}")

        parts.append("=" * 50)
        parts.append(
            "Please provide detailed assistance based on the above analysis.\n"
        )

        return "\n".join(parts)

    def analyze_simple(self, image: Union[str, Path, bytes, "PILImage"]) -> str:
        """
        Quick analysis returning just the key findings as a string.

        Useful for simple integrations where you just need a text summary.

        Args:
            image: Image in any supported format

        Returns:
            String summary of analysis, or error message
        """
        result = self.analyze(image)

        if not result.success:
            return f"Analysis failed: {result.error}"

        lines = [f"Screenshot type: {result.screenshot_type}"]

        if result.extracted_text:
            lines.append(f"Text found: {result.extracted_text[:200]}...")

        if result.error_codes:
            lines.append(f"Error codes: {', '.join(result.error_codes)}")

        lines.append(f"Analysis: {result.analysis}")

        return "\n".join(lines)


# Singleton instance
_analyzer: VisionAnalyzer | None = None
_analyzer_lock = threading.Lock()


def get_vision_analyzer() -> VisionAnalyzer:
    """
    Get or create singleton VisionAnalyzer instance (thread-safe).

    Returns:
        Global VisionAnalyzer instance
    """
    global _analyzer
    if _analyzer is None:
        with _analyzer_lock:
            if _analyzer is None:
                _analyzer = VisionAnalyzer()
    return _analyzer


def reset_vision_analyzer() -> None:
    """Reset the global VisionAnalyzer instance (useful for testing)."""
    global _analyzer
    with _analyzer_lock:
        _analyzer = None
