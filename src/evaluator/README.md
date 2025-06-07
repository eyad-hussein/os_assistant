# OS Assistant Evaluation System



## Run evaluation on all samples
uv run .\src\evaluator\run_evaluation.py --dataset datasets/file_system_dataset.json --batch-size 5

## Run evaluation on a specific range of samples (samples 5-15)
uv run .\src\evaluator\run_evaluation.py --start 5 --end 15

## Continue an existing evaluation
uv run .\src\evaluator\run_evaluation.py --continue-from evaluation_results/evaluation_20230606_120000.json

## Run with smaller batch size for more frequent updates
uv run .\src\evaluator\run_evaluation.py--batch-size 2

## Run evaluation on second half of dataset
uv run .\src\evaluator\run_evaluation.py --start 20 

## Run evaluation on just the first 10 samples
uv run .\src\evaluator\run_evaluation.py --end 9
