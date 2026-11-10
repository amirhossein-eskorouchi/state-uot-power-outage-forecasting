# Reproducibility guide

## 1. Inputs and expected directory structure

The final Python script is intentionally **self-contained**, with the validated feature engineering and model implementation embedded in it. It resolves input paths relative to its own location.

Obtain the official challenge files from an authorized source and place them in the repository root, next to the script:

```text
state-uot-power-outage-forecasting/
├── FINAL_TRIAL22_END_TO_END.py
├── requirements.txt
├── DM_Train.csv              # required; not included in Git
├── DM_Test.csv               # required; not included in Git
└── sample_submission.csv     # required; not included in Git
```

**Do not publish the CSV files without confirming redistribution permission.** The repository's `.gitignore` excludes these files by default.

## 2. Python environment

The originally supplied pinned dependencies are reproduced verbatim in `requirements.txt`:

```text
numpy==2.0.2
pandas==2.2.2
scikit-learn==1.6.1
scipy==1.13.1
torch==2.6.0
geomloss==0.3.1
matplotlib==3.9.4
```

Python 3.11 is a reasonable starting environment for these requirements, but it has **not been independently tested as part of this repository packaging**. Torch GPU support requires an environment-specific compatible installation; the generic pip command below may not select the desired CUDA build.

### Windows PowerShell / VS Code terminal

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell execution policy prevents activation, use `& .\\.venv\\Scripts\\python.exe` in place of `python` without activating the virtual environment, or consult your institution's policy. Do not weaken security settings merely to run this code.

## 3. Full-data training and prediction

From the repository root:

```powershell
python FINAL_TRIAL22_END_TO_END.py
```

The original script trains **five full-data models** using actual training seeds:

```text
10000, 10017, 10042, 10073, 10123
```

It builds the final averaged submission and saves diagnostics, model artifacts, and provenance under:

```text
final_trial22_state_uot_submission/
```

The expected official output path is:

```text
final_trial22_state_uot_submission/competition_ready/prediction_file.csv
```

The script also writes a reproducibility `code.zip` and a run log in its output directory. Training can require significant GPU memory and computation; it is **not necessary** to retrain the model just to browse the source code or documentation.

## 4. Validation and provenance

The submission script checks, among other things:

- Required identifiers and row order against the official template
- Expected missing targets beyond the available time window
- Nonnegative OSI predictions
- Physical-target-time consistency among repeated requests
- Per-model and ensemble prediction diagnostics

The final script contains frozen hyperparameters and records model/source information. Optional earlier experiment directories are used for provenance copies **only if they are present**; they are not required by the final training path.

This repository includes the *final submission* code only. It is not a complete archive of every model-selection experiment or five-fold validation artifact. The related methodology is summarized in [METHODOLOGY.md](METHODOLOGY.md).

## 5. What is omitted from version control

The official CSVs, local virtual environment, generated model checkpoints, output predictions, manifests containing local file paths, logs, and previous experiment directories are excluded by `.gitignore`. A file being ignored locally does not mean it is safe to disclose elsewhere. Inspect the staging set with `git status` before publishing.

## 6. Check without training

For a quick source syntax check:

```powershell
python -m py_compile FINAL_TRIAL22_END_TO_END.py
```

This compiles syntax but does **not** install dependencies, load data, run a model, or reproduce competition scores.
