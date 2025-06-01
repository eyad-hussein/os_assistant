# OS Assistant Evaluation System

This system evaluates the quality of OS Assistant responses by comparing them against expected responses using LLM-based evaluation.

## File Overview

### `/src/os_assistant/prompts/evaluation.yaml`
Contains the evaluation prompts used by the LLM judge to assess OS Assistant responses. There are specialized prompts for different query types (command, information) and a general fallback prompt.

### `/src/evaluator/utils/parser.py`
Parses and validates LLM evaluation responses, extracting the JSON structure with scores and reasoning.

### `/src/evaluator/utils/models.py`
Defines Pydantic models for evaluation scores, results, and dataset structures, ensuring type safety and validation.

### `/src/evaluator/run_evaluation.py`
Main script to execute the evaluation process, load datasets, run evaluation, and save results.

### `/src/evaluator/core/llm_judge.py`
Uses LLMs to evaluate OS Assistant responses against expected responses, applying the appropriate prompt templates.

### `/src/evaluator/core/evaluator.py`
Core evaluation logic that processes samples, generates summaries, and aggregates results across different dimensions.

## Modifying Evaluation Criteria

If you want to change the evaluation criteria, you need to modify multiple files in a coordinated way:

1. **Primary: `/src/os_assistant/prompts/evaluation.yaml`**
   - This is where you define the actual criteria in the prompts
   - Each prompt template includes the specific criteria to evaluate
   - The JSON structure expected in the response must match the models

2. **Secondary: `/src/evaluator/utils/models.py`**
   - Update the score models (e.g., `CommandScores`, `InformationScores`, `GeneralScores`)
   - Add, remove, or modify score fields with appropriate validation
   - Ensure the `EvaluationResult` model can handle the updated score structures

3. **Secondary: `/src/evaluator/utils/parser.py`**
   - Update the `determine_score_type` function if you've added new score types
   - Make sure the parsing logic can handle any new JSON structure

## Example: Adding a New Evaluation Criterion

Let's say you want to add a "Helpfulness" criterion to command evaluation:

1. In `evaluation.yaml`, add "Helpfulness" to the command evaluation criteria list and update the JSON structure example
2. In `models.py`, add `helpfulness: int = Field(..., ge=1, le=5)` to the `CommandScores` class
3. In `parser.py`, update the `command_keys` set in `determine_score_type` to include "helpfulness"
