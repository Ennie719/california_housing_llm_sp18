# California Housing Prediction

This project trains and evaluates a Random Forest regression model for the
California Housing dataset and tracks whether incoming monthly data differs
from the reference data used for monitoring.

## What The Project Does

- Loads the California Housing dataset with scikit-learn.
- Validates and splits the data into training and validation sets.
- Trains a configurable `RandomForestRegressor`.
- Logs model parameters and MSE, RMSE, and R² metrics to MLflow.
- Compares reference data with Month 1, Month 2, and Month 3 data using
	Evidently.
- Writes drift results and monitoring reports to a `reports/` directory.

## Repository Layout

```text
california_housing.py       Main MLflow training entry point
src/                        Preprocessing, training, and evaluation modules
configs/config.yaml         Dataset and model configuration
MLproject                   MLflow Project definition
drift-monitoring/           Monitoring scripts and sample data
src/drift-monitoring/       Compatible monitoring workflow used below
notebooks/                  Exploratory scripts and notebooks
tests/                      Project tests
requirements.txt            Pinned Python dependencies
```

The repository contains older duplicate monitoring scripts in
`drift-monitoring/`. The commands below use `src/drift-monitoring/`, which is
the copy aligned with the pinned Evidently 0.4.x API.

## Requirements

- macOS, Linux, or Windows
- Python 3.11
- pip

Python 3.11 is important because the dependency set pins NumPy 1.26.4, while
the older Evidently version is incompatible with NumPy 2.x and Python 3.13.

## Setup

From the project root:

```bash
python3.11 -m venv .venv311
source .venv311/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, activate the environment with:

```powershell
.venv311\Scripts\Activate.ps1
```

Verify the important dependency versions:

```bash
python -c "import numpy, evidently; print(numpy.__version__)"
```

The expected NumPy version is `1.26.4`.

## Train The Model

Run the main training workflow from the project root:

```bash
source .venv311/bin/activate
python california_housing.py
```

The script reads `configs/config.yaml`, trains the configured random forest,
prints evaluation metrics, and logs the run to the local MLflow tracking store.

The same workflow can be launched through MLflow:

```bash
mlflow run .
```

To inspect recorded runs in the MLflow UI:

```bash
mlflow ui
```

Then open the local URL printed by MLflow.

## Run Drift Monitoring

The monitoring scripts expect to be run from their own directory because the
CSV paths are relative to that directory:

```bash
cd src/drift-monitoring
python drift_check.py reference_data.csv month1_data.csv
python drift_check.py reference_data.csv month2_data.csv
python drift_check.py reference_data.csv month3_data.csv
python drift_over_time.py
```

The scripts create:

- `src/drift-monitoring/reports/drift_check_result.json`
- `src/drift-monitoring/reports/drift_timeline.json`
- `src/drift-monitoring/reports/drift_month1.html`
- `src/drift-monitoring/reports/drift_month2.html`
- `src/drift-monitoring/reports/drift_month3.html`

To generate the HTML reports, run:

```bash
python detect_drift.py
```

## Current Drift Findings

For the Month 2 comparison, four of nine features exceeded Evidently's drift
threshold of `0.1`:

| Feature | Interpretation |
| --- | --- |
| `AveBedrms` | Change in the average number of bedrooms per household |
| `HouseAge` | Change in the housing-age distribution |
| `MedInc` | Change in the median-income distribution |
| `Population` | Change in the population distribution |

The other five features remained below the threshold. The drift share was
44.4%, which triggered the critical threshold of 40%. This sets the status to
`critical`, so `drift_check.py` exits with a non-zero status. If this check is
run in CI, the pipeline will therefore fail until the drift is investigated.
The dataset-level drift flag was `False`.

### Expected Impact And Action

This drift could affect model performance. `MedInc` is likely an important
predictor of house value, and a large distribution change can move new inputs
away from the training population. The other drifted features may also matter
if their relationships with house value change. Drift alone does not prove
that model accuracy has declined.

The recommended action is to investigate and continue monitoring. Check for
data-pipeline issues and evaluate the model on newly labeled data using RMSE
and R². Retrain if performance has degraded, feature-target relationships
have changed, or the drift persists in later monitoring periods.
