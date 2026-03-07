# Product Hunt Launch Success Predictor

Machine learning pipeline to predict Product Hunt launch performance using launch-time metadata and text features.

## Setup

1. Create a virtual environment and install dependencies:
   - `python3 -m venv .venv`
   - `source .venv/bin/activate`
   - `python -m pip install --upgrade pip`
   - `python -m pip install -r requirements.txt`
2. Install OpenMP runtime on macOS (required by XGBoost):
   - `brew install libomp`
3. Ensure `.env` includes:
   - `PH_TOKEN`
   - `POSTED_AFTER`
   - `POSTED_BEFORE`
   - `PAGE_SIZE`
   - `MAX_POSTS`
   - `RANDOM_STATE`

## Run

- Full pipeline: `python src/pipeline.py`
- Individual stages:
  - `python src/pipeline.py --stage collect`
  - `python src/pipeline.py --stage preprocess`
  - `python src/pipeline.py --stage features`
  - `python src/pipeline.py --stage train`
  - `python src/pipeline.py --stage evaluate`

## Outputs

- `data/raw_posts.csv`: raw Product Hunt launch data
- `data/processed_posts.csv`: processed data with labels/features
- `data/features/`: saved model feature matrices
- `results/`: metrics, plots, and serialized models
