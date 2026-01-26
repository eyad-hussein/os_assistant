from os_assistant.core.state import AssistantState
from os_assistant.tools.vision import VisionAnalysisResult, get_vision_analyzer
from os_assistant.utils import LOGGER
from os_assistant.utils.settings import VISION_ENABLED


def _test_vision_model(analyzer) -> bool:
    """
    Test if the vision model is accessible.

    Args:
        analyzer: VisionAnalyzer instance

    Returns:
        True if model is accessible, False otherwise
    """
    try:
        # Try to access the model property (lazy-loads it)
        _ = analyzer.model
        return True
    except Exception as e:
        LOGGER.debug(f"Vision model test failed: {e}")
        return False


def vision_analysis_node(state: AssistantState) -> AssistantState:
    """
    Analyze attached images and enhance the user prompt.

    This node:
    1. Checks if an image is attached to the query
    2. Analyzes the image using a vision model (LLaVA, llama3.2-vision)
    3. Enhances the prompt with extracted information (text, errors, context)
    4. Stores analysis results in state for later use

    The enhanced prompt includes:
    - Extracted text from the image
    - Detected error codes
    - Screenshot type classification
    - Preliminary analysis and suggestions

    Args:
        state: Current assistant state

    Returns:
        Updated state with vision analysis results
    """
    LOGGER.info("\n" + "=" * 50)
    LOGGER.info("NODE: vision_analysis_node")
    LOGGER.info("=" * 50)

    # Check if vision is globally enabled
    if not VISION_ENABLED:
        LOGGER.debug("Vision analysis is disabled in settings. Skipping.")
        return state

    # Check if image is present in state
    image_data = state.get("attached_image")

    if not image_data:
        LOGGER.debug("No image attached to query. Skipping vision analysis.")
        return state

    # Get the vision analyzer
    analyzer = get_vision_analyzer()

    if not analyzer.is_enabled():
        LOGGER.warning(
            "Vision analyzer is not available. "
            "Check if Pillow is installed: pip install Pillow"
        )
        state["vision_analysis"] = {"error": "Vision analyzer not available"}
        return state

    # Test if vision model is accessible before proceeding
    try:
        test_result = _test_vision_model(analyzer)
        if not test_result:
            LOGGER.warning(
                "Vision model is not accessible. "
                "Ensure Ollama is running and the vision model is installed."
            )
            state["vision_analysis"] = {
                "error": "Vision model not accessible. Check Ollama server."
            }
            return state
    except Exception as e:
        LOGGER.warning(f"Failed to test vision model availability: {e}")
        state["vision_analysis"] = {"error": f"Vision model test failed: {e}"}
        return state

    LOGGER.info("Image detected. Performing vision analysis...")

    try:
        # Get the original prompt for context
        original_prompt = state.get("prompt", "")

        # Analyze the image
        result: VisionAnalysisResult = analyzer.analyze(
            image=image_data, user_prompt=original_prompt
        )

        if result.success:
            # Store original prompt before enhancement
            if original_prompt:
                state["original_prompt"] = original_prompt

            # Enhance prompt with vision analysis
            state["prompt"] = result.enhanced_prompt

            # Store detailed analysis for potential use by other nodes
            state["vision_analysis"] = {
                "success": True,
                "extracted_text": result.extracted_text,
                "error_codes": result.error_codes,
                "screenshot_type": result.screenshot_type,
                "analysis": result.analysis,
                "suggested_actions": result.suggested_actions,
            }

            # Log summary
            LOGGER.info("Vision analysis complete:")
            LOGGER.info(f"   Screenshot type: {result.screenshot_type}")
            LOGGER.info(f"   Error codes found: {len(result.error_codes)}")
            if result.error_codes:
                LOGGER.info(f"   Codes: {', '.join(result.error_codes[:5])}")
            LOGGER.info(f"   Suggested actions: {len(result.suggested_actions)}")

            # Log enhanced prompt preview (truncated)
            if len(result.enhanced_prompt) > 200:
                LOGGER.debug(
                    f"Enhanced prompt (preview): {result.enhanced_prompt[:200]}..."
                )
            else:
                LOGGER.debug(f"Enhanced prompt: {result.enhanced_prompt}")

        else:
            LOGGER.warning(f"Vision analysis failed: {result.error}")
            state["vision_analysis"] = {
                "success": False,
                "error": result.error,
            }

    except Exception as e:
        LOGGER.error(f"Vision analysis error: {e}")
        state["vision_analysis"] = {
            "success": False,
            "error": str(e),
        }

    return state


def is_vision_available() -> bool:
    """
    Check if vision analysis is available and properly configured.

    Returns:
        True if vision can be used, False otherwise
    """
    if not VISION_ENABLED:
        return False

    try:
        analyzer = get_vision_analyzer()
        return analyzer.is_enabled()
    except Exception:
        return False
