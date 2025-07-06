# OS Assistant Dataset Generator

A tool for generating question-answer datasets for OS Assistant training.

## Components

### Main Files
- `run_dataset_generator.py`: Main entry point, controls generation process
- `generate_dataset.py`: Core functions for dataset creation
- `log_sampler.py`: Retrieves system logs with sequential connections
- `question_generator.py`: Creates diverse question types from logs
- `similarity_checker.py`: Prevents duplicate questions

### Question Types
- **Log-based**: Generated from actual system logs
- **Random**: General OS questions not tied to specific logs
- **Code Execution**: Questions requiring code to answer
