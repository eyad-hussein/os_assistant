import os
from dataclasses import dataclass

# Import from settings to maintain consistency
from dagent.utils.settings import MODEL_BASE_URL


@dataclass
class VisionConfig:
    """
    Configuration for vision analysis.

    Attributes:
        enabled: Whether vision analysis is enabled
        model_name: Ollama vision model to use (e.g., 'llava:7b', 'llama3.2-vision:11b')
        base_url: Ollama server URL for vision model
        max_image_size: Maximum image dimension in pixels (resized if larger)
        jpeg_quality: JPEG compression quality (1-100) for processed images
        supported_formats: Tuple of supported image file extensions
        timeout_seconds: Timeout for vision model inference
    """

    enabled: bool = True
    model_name: str = "llava:7b"
    base_url: str = "http://localhost:11434"
    max_image_size: int = 1024
    jpeg_quality: int = 85
    supported_formats: tuple = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp")
    timeout_seconds: int = 60

    def __post_init__(self):
        """Apply environment overrides after initialization."""
        # Allow base_url from settings if available
        if MODEL_BASE_URL:
            self.base_url = MODEL_BASE_URL

    @classmethod
    def from_env(cls) -> "VisionConfig":
        """
        Create configuration from environment variables.

        Environment variables:
            VISION_ENABLED: 'true' or 'false' (default: 'true')
            VISION_MODEL: Model name (default: 'llava:7b')
            MODEL_BASE_URL: Ollama server URL (default: 'http://localhost:11434')
            VISION_MAX_IMAGE_SIZE: Max dimension in pixels (default: 1024)
            VISION_TIMEOUT: Timeout in seconds (default: 60)

        Returns:
            VisionConfig instance with values from environment
        """
        return cls(
            enabled=os.getenv("VISION_ENABLED", "true").lower() == "true",
            model_name=os.getenv("VISION_MODEL", "llava:7b"),
            base_url=os.getenv("MODEL_BASE_URL", "http://localhost:11434"),
            max_image_size=int(os.getenv("VISION_MAX_IMAGE_SIZE", "1024")),
            timeout_seconds=int(os.getenv("VISION_TIMEOUT", "60")),
        )

    def validate(self) -> list[str]:
        """
        Validate configuration values.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        if not self.model_name:
            errors.append("Vision model name cannot be empty")

        if not self.base_url:
            errors.append("Vision base URL cannot be empty")

        if self.max_image_size < 100:
            errors.append("Max image size must be at least 100 pixels")

        if not 1 <= self.jpeg_quality <= 100:
            errors.append("JPEG quality must be between 1 and 100")

        if self.timeout_seconds < 1:
            errors.append("Timeout must be at least 1 second")

        return errors


# Default configuration singleton
VISION_CONFIG = VisionConfig.from_env()
