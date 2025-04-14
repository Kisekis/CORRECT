# Project Structure

## Directory Overview
- `CIVDataset/`: Contains 2000 collected vulnerability pairs
- `load_history/CIVDataset-Small.json`: Contains 400 sampled vulnerability pairs
- `data/`: Contains meta data
- `results/`: Contains raw results

# Dataset Construction

## Prerequisites
- Install `cflow` and `joern`
- Run `pip install -r requirements.txt`
- Clone required repositories into the `projects` folder

## Usage
Run `python main.py` to collect all contexts.

Results (including final results and intermediate CPG outputs) will be saved in:
- `CIVDataset`
- `cpgs`

# Analysis

## Prerequisites
Add API key and models in `program/analyze/get_analyzer.py`

## Usage
1. `cd program`
2. Run: `python3 main.py --mode {mode} --dataset CIVDataset-Small --detection_model {model}`

### Available Modes
- `no_context`: Corresponds to w/o context, w/o revision
- `context`: Corresponds to w/context, w/o revision
- `normal`: Corresponds to Lenient Mode
- `hard`: Corresponds to Strict Mode

# Evaluation Results (RQ1-RQ3)

## Usage
1. `cd program`
2. Run: `python3 evaluator.py --evaluator {evaluator}`

### Available Evaluators
- `base`: Fig 5
- `abnormal`: Appendix - Abnormal output of LLMs
- `f1`: Fig 4
- `cwe_scaling`: Fig 6
- `feedback`: Appendix - Feedback settings in Strict Mode
- `scaling`: Fig 7

### Results
The evaluation results will be saved in `program/metrics_plots` directory.
