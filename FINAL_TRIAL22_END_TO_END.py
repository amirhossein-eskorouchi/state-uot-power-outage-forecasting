from __future__ import annotations

"""
PHASE 14 -- FINAL FULL-DATA TRIAL-22 STATE-UOT TRAINING AND SUBMISSION
=====================================================================

FINAL MODEL-SELECTION STATUS
----------------------------
The Stage-1 Optuna search selected Trial 22. A subsequent five-seed x five-fold
robustness audit found Trial 22 and the previous State-UOT incumbent essentially
tied on the competition-aligned RMSE criterion, while Trial 22 retained lower MAE
across all four horizons and better short-horizon RMSE. The team selected Trial 22
for final deployment and froze it before this run.

NO further model selection, hyperparameter search, architecture tuning, blending,
or use of test labels occurs in this script.

TRAINING
--------
Use ALL 239 training counties.

Architecture:
    exact FinalCNNV2 from the embedded validated base pipeline

Point loss:
    masked SmoothL1 on log1p(OSI)

Distributional regularizer:
    state-conditioned categorical unbalanced Sinkhorn divergence

FROZEN TRIAL-22 SETTINGS:
    reach       = 1.9492145906383787
    blur        = 0.11988343825386838
    diameter    = sqrt(2)
    lambda_uot  = 0.0012262539110903918
    warm-up     = 20 epochs
    ramp        = 30 epochs
    epochs      = 120
    batch size  = 16
    Adam lr     = 0.0009448516088066339

Five full-data models:
    base seeds = [0, 17, 42, 73, 123]
    actual full-data training seed = base_seed + 10000

FINAL PREDICTION
----------------
Each full-data seed predicts the official test set. The competition prediction is
the simple average of the five seed submissions.

OUTPUT
------
final_trial22_state_uot_submission/

The package saves:
    * exact full-data scalers
    * five model checkpoints
    * five training histories
    * five raw test prediction arrays
    * five seed submissions
    * five-seed averaged final submission
    * competition_ready/prediction_file.csv
    * competition_ready/code.zip
    * strict submission validation
    * target-consistency checks
    * per-row/per-horizon seed disagreement
    * test prediction distribution summaries
    * optional comparison to the previous EXP05 test submission
    * training / uncertainty / prediction-distribution visualizations
    * top predicted-peak county trajectories
    * source/environment/hash manifests
    * console log
    * artifact inventory

IMPORTANT
---------
There is no test ground truth in this stage. Any comparison between test
prediction files is prediction-difference analysis, NOT a performance comparison.

The competition requires a prediction file, reproducible code, and a written
report. This single self-contained script creates the prediction file and a reproducible code.zip.
The written report is prepared separately.
"""

import hashlib
import json
import math
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import pickle
import platform
import shutil
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

try:
    import geomloss
    from geomloss import SamplesLoss
except Exception as exc:
    raise ImportError(
        "GeomLoss 0.3.1 is required in the active Spyder environment."
    ) from exc

# ---------------------------------------------------------------------
# EMBEDDED VALIDATED BASE PIPELINE
# ---------------------------------------------------------------------
# The exact helper pipeline used during model development is embedded below
# so this file is fully self-contained. It is executed inside its own module
# namespace so its globals remain isolated from the final Trial-22 globals.
import types

EMBEDDED_BASE_SOURCE = '# ================================================================\n# final_reproducible_pipeline.py\n# INFORMS 2026 Data Mining Society Data Challenge\n#\n# SUPPORT MODULE FOR THE FINAL TRIAL-22 SUBMISSION\n# ------------------------------------------------\n# This file contains the validated data engineering, FinalCNNV2 architecture,\n# prediction construction, and strict submission-validation utilities used by\n# phase14_final_trial22_submission.py.\n#\n# FINAL ENTRY POINT:\n#     python phase14_final_trial22_submission.py\n#\n# Do not run this support module directly for the final submission.\n# ================================================================\n\nimport os\nos.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")\n\nfrom pathlib import Path\nfrom datetime import datetime\nimport argparse\nimport hashlib\nimport json\nimport platform\nimport random\nimport sys\nimport time\nimport traceback\n\nimport numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\n\nfrom scipy.stats import pearsonr\nfrom sklearn.model_selection import StratifiedGroupKFold\nfrom sklearn.preprocessing import StandardScaler\n\nimport torch\nimport torch.nn as nn\nfrom torch.utils.data import Dataset, DataLoader\n\n\n# ================================================================\n# 1. PROJECT CONFIGURATION\n# ================================================================\n\nROOT = Path(__file__).resolve().parent\n\nTRAIN_PATH = ROOT / "DM_Train.csv"\nTEST_PATH = ROOT / "DM_Test.csv"\nTEMPLATE_PATH = ROOT / "sample_submission.csv"\n\nSPLIT_DIR = ROOT / "splits"\nFOLD_PATH = SPLIT_DIR / "county_folds_seed42.csv"\n\nFINAL_ROOT = ROOT / "final_artifacts"\nCONFIG_DIR = FINAL_ROOT / "config"\nLOG_DIR = FINAL_ROOT / "logs"\nMETRICS_DIR = FINAL_ROOT / "metrics"\nOOF_DIR = FINAL_ROOT / "oof"\nMODEL_DIR = FINAL_ROOT / "models"\nSUBMISSION_DIR = FINAL_ROOT / "submissions"\nFIGURE_DIR = FINAL_ROOT / "figures"\nENV_DIR = FINAL_ROOT / "environment"\n\nfor p in [\n    SPLIT_DIR,\n    FINAL_ROOT,\n    CONFIG_DIR,\n    LOG_DIR,\n    METRICS_DIR,\n    OOF_DIR,\n    MODEL_DIR,\n    SUBMISSION_DIR,\n    FIGURE_DIR,\n    ENV_DIR,\n]:\n    p.mkdir(parents=True, exist_ok=True)\n\n# Competition / validation constants\nCUTOFF = 72\nN_HOURS = 216\nN_FUTURE = 144\nN_FOLDS = 5\nFOLD_SEED = 42\nHORIZONS = [1, 6, 24, 48]\n\nTARGET_COLS = [\n    "osi_target_t01h",\n    "osi_target_t06h",\n    "osi_target_t24h",\n    "osi_target_t48h",\n]\n\nEXPECTED_TARGET_N = {\n    1: 239 * 143,\n    6: 239 * 138,\n    24: 239 * 120,\n    48: 239 * 96,\n}\n\nEXPECTED_TEMPLATE_NANS = {\n    "osi_target_t01h": 63,\n    "osi_target_t06h": 378,\n    "osi_target_t24h": 1512,\n    "osi_target_t48h": 3024,\n}\n\n# Final five seeds.\nFINAL_SEEDS = [0, 17, 42, 73, 123]\n\n# Final EXP05 hyperparameters.\nFINAL_EPOCHS = 120\nFINAL_BATCH_SIZE = 16\nFINAL_LR = 1e-3\n\n# Baseline EXP03 hyperparameters, matching the cleaned reproduction.\nBASELINE_SEED = 42\nBASELINE_WINDOW = 24\nBASELINE_EPOCHS = 15\nBASELINE_BATCH_SIZE = 256\nBASELINE_LR = 1e-3\n\nBOOTSTRAP_REPS = 5000\nBOOTSTRAP_SEED = 20260912\n\nDEVICE = torch.device(\n    "cuda" if torch.cuda.is_available() else "cpu"\n)\n\nORIGINAL_BASELINE_URL = (\n    "https://github.com/ezy-8/INFORMS2026DataChallenge/"\n    "tree/main/code%20(Pias%20%2B%20Munirul)"\n)\n\n\n# ================================================================\n# 2. FEATURES\n# ================================================================\n\nWEATHER = [\n    "gust",\n    "wind_speed_10m",\n    "t2m",\n    "d2m",\n    "sp",\n    "mslma",\n    "blh",\n    "tp",\n    "rain",\n    "csnow",\n    "sdwe",\n    "tcc",\n    "lcc",\n    "mcc",\n    "hcc",\n    "sdswrf",\n    "direct_rad",\n    "diffuse_rad",\n    "r2",\n    "vpd",\n    "et0",\n    "soil_moist",\n]\n\nOUTAGE_STATE = [\n    "osi",\n    "P_t",\n    "N_t",\n    "D_t",\n    "R_t",\n    "outage_pct",\n]\n\nFUTURE_COLS = (\n    WEATHER\n    + [\n        "wind_sin",\n        "wind_cos",\n    ]\n)\n\nHISTORY_COLS = (\n    FUTURE_COLS\n    + OUTAGE_STATE\n)\n\nBASELINE_STATIC = [\n    "gust_ge36",\n    "gust_ge48",\n]\n\n\n# ================================================================\n# 3. LOGGING / REPRODUCIBILITY\n# ================================================================\n\nclass Tee:\n\n    def __init__(self, *streams):\n        self.streams = streams\n\n    def write(self, text):\n        for stream in self.streams:\n            stream.write(text)\n            stream.flush()\n\n    def flush(self):\n        for stream in self.streams:\n            stream.flush()\n\n\ndef sha256_file(path, chunk_size=1024 * 1024):\n\n    sha = hashlib.sha256()\n\n    with open(path, "rb") as f:\n\n        while True:\n            chunk = f.read(chunk_size)\n\n            if not chunk:\n                break\n\n            sha.update(chunk)\n\n    return sha.hexdigest()\n\n\ndef set_seed(seed):\n\n    random.seed(seed)\n    np.random.seed(seed)\n\n    torch.manual_seed(seed)\n\n    if torch.cuda.is_available():\n        torch.cuda.manual_seed_all(seed)\n\n\ndef configure_determinism():\n\n    if torch.cuda.is_available():\n\n        torch.backends.cudnn.benchmark = False\n        torch.backends.cudnn.deterministic = True\n\n        torch.backends.cuda.matmul.allow_tf32 = False\n        torch.backends.cudnn.allow_tf32 = False\n\n    try:\n        torch.use_deterministic_algorithms(\n            True,\n            warn_only=True,\n        )\n    except TypeError:\n        torch.use_deterministic_algorithms(True)\n\n\ndef write_environment_snapshot():\n\n    info = {\n        "created": datetime.now().isoformat(),\n        "python": sys.version,\n        "platform": platform.platform(),\n        "numpy": np.__version__,\n        "pandas": pd.__version__,\n        "torch": torch.__version__,\n        "device": str(DEVICE),\n        "gpu": (\n            torch.cuda.get_device_name(0)\n            if torch.cuda.is_available()\n            else None\n        ),\n    }\n\n    try:\n        import sklearn\n        info["scikit_learn"] = sklearn.__version__\n    except Exception:\n        info["scikit_learn"] = None\n\n    try:\n        import scipy\n        info["scipy"] = scipy.__version__\n    except Exception:\n        info["scipy"] = None\n\n    try:\n        import matplotlib\n        info["matplotlib"] = matplotlib.__version__\n    except Exception:\n        info["matplotlib"] = None\n\n    path = ENV_DIR / "environment.json"\n\n    with open(path, "w", encoding="utf-8") as f:\n        json.dump(info, f, indent=2)\n\n    return path\n\n\ndef write_input_manifest():\n\n    files = [\n        TRAIN_PATH,\n        TEST_PATH,\n        TEMPLATE_PATH,\n    ]\n\n    if FOLD_PATH.exists():\n        files.append(FOLD_PATH)\n\n    rows = []\n\n    for path in files:\n\n        rows.append(\n            {\n                "file": str(path.relative_to(ROOT)),\n                "size_bytes": path.stat().st_size,\n                "sha256": sha256_file(path),\n            }\n        )\n\n    out = pd.DataFrame(rows)\n\n    path = ENV_DIR / "input_manifest.csv"\n\n    out.to_csv(path, index=False)\n\n    return path\n\n\ndef write_run_config(args):\n\n    config = {\n        "created": datetime.now().isoformat(),\n        "mode": args.mode,\n        "force": bool(args.force),\n        "root": str(ROOT),\n        "cutoff_hour_index": 71,\n        "first_forecast_origin_index": 72,\n        "horizons": HORIZONS,\n        "fold_seed": FOLD_SEED,\n        "final_model": (\n            "EXP05 CNN-v2 fixed 72-hour genuine history + "\n            "144-hour future weather, five-seed average"\n        ),\n        "final_seeds": FINAL_SEEDS,\n        "final_epochs": FINAL_EPOCHS,\n        "final_batch_size": FINAL_BATCH_SIZE,\n        "final_learning_rate": FINAL_LR,\n        "baseline": (\n            "Cleaned/leakage-safe reproduction of the initial "\n            "Pias + Munirul moving-window CNN-LSTM (EXP03)"\n        ),\n        "baseline_source_url": ORIGINAL_BASELINE_URL,\n        "baseline_seed": BASELINE_SEED,\n        "baseline_window": BASELINE_WINDOW,\n        "baseline_epochs": BASELINE_EPOCHS,\n        "script_sha256": sha256_file(Path(__file__)),\n    }\n\n    path = CONFIG_DIR / "final_pipeline_config.json"\n\n    with open(path, "w", encoding="utf-8") as f:\n        json.dump(config, f, indent=2)\n\n    return path\n\n\n# ================================================================\n# 4. DATA LOADING / AUDIT\n# ================================================================\n\ndef require_inputs():\n\n    for path in [\n        TRAIN_PATH,\n        TEST_PATH,\n        TEMPLATE_PATH,\n    ]:\n        if not path.exists():\n            raise FileNotFoundError(\n                f"Required input file is missing:\\n{path}"\n            )\n\n\ndef engineer_data(df):\n\n    df = (\n        df\n        .sort_values(\n            [\n                "fipsCode",\n                "timestamp_et",\n            ]\n        )\n        .reset_index(drop=True)\n        .copy()\n    )\n\n    df["hour_idx"] = (\n        df\n        .groupby("fipsCode")\n        .cumcount()\n    )\n\n    radians = (\n        2.0\n        * np.pi\n        * df["wind_dir_10m"].astype(float)\n        / 360.0\n    )\n\n    df["wind_sin"] = np.sin(radians).astype(np.float32)\n    df["wind_cos"] = np.cos(radians).astype(np.float32)\n\n    df["gust_ge36"] = (\n        df["gust"] >= 36\n    ).astype(np.float32)\n\n    df["gust_ge48"] = (\n        df["gust"] >= 48\n    ).astype(np.float32)\n\n    return df\n\n\ndef load_data():\n\n    train = pd.read_csv(\n        TRAIN_PATH,\n        parse_dates=["timestamp_et"],\n    )\n\n    test = pd.read_csv(\n        TEST_PATH,\n        parse_dates=["timestamp_et"],\n    )\n\n    # Reconstruct test OSI only where the official outage-state\n    # components are observed. After the cutoff these fields are NaN,\n    # so reconstructed OSI is also NaN.\n    test["osi"] = np.maximum(\n        0.0,\n        0.40 * test["P_t"]\n        + 0.35 * test["N_t"]\n        + 0.25 * test["D_t"]\n        - 0.10 * test["R_t"]\n    )\n\n    return (\n        engineer_data(train),\n        engineer_data(test),\n    )\n\n\ndef audit_inputs(train, test):\n\n    print("\\n" + "=" * 78)\n    print("INPUT / LEAKAGE AUDIT")\n    print("=" * 78)\n\n    assert len(train) == 51624\n    assert len(test) == 13608\n\n    assert train["fipsCode"].nunique() == 239\n    assert test["fipsCode"].nunique() == 63\n\n    assert (\n        set(train["fipsCode"])\n        .isdisjoint(\n            set(test["fipsCode"])\n        )\n    )\n\n    train_count = (\n        train\n        .groupby("fipsCode")\n        .size()\n    )\n\n    test_count = (\n        test\n        .groupby("fipsCode")\n        .size()\n    )\n\n    assert (\n        train_count.min()\n        == train_count.max()\n        == N_HOURS\n    )\n\n    assert (\n        test_count.min()\n        == test_count.max()\n        == N_HOURS\n    )\n\n    cutoff_timestamp = pd.Timestamp(\n        "2026-03-13 23:00:00"\n    )\n\n    cutoff_rows = (\n        test[\n            test["hour_idx"] == 71\n        ]\n    )\n\n    assert (\n        cutoff_rows["timestamp_et"]\n        == cutoff_timestamp\n    ).all()\n\n    hidden = (\n        test["hour_idx"]\n        >= CUTOFF\n    )\n\n    for col in [\n        "outageCount",\n        "outage_pct",\n        "P_t",\n        "N_t",\n        "D_t",\n        "R_t",\n    ]:\n\n        assert (\n            test.loc[\n                hidden,\n                col,\n            ]\n            .notna()\n            .sum()\n            == 0\n        )\n\n    assert (\n        train[WEATHER]\n        .isna()\n        .sum()\n        .sum()\n        == 0\n    )\n\n    assert (\n        test[WEATHER]\n        .isna()\n        .sum()\n        .sum()\n        == 0\n    )\n\n    template = pd.read_csv(\n        TEMPLATE_PATH\n    )\n\n    assert template.shape == (9072, 8)\n\n    for col, expected in (\n        EXPECTED_TEMPLATE_NANS.items()\n    ):\n\n        assert (\n            template[col]\n            .isna()\n            .sum()\n            == expected\n        )\n\n    rows = [\n        {\n            "check": "train_rows",\n            "value": len(train),\n        },\n        {\n            "check": "test_rows",\n            "value": len(test),\n        },\n        {\n            "check": "train_counties",\n            "value": train["fipsCode"].nunique(),\n        },\n        {\n            "check": "test_counties",\n            "value": test["fipsCode"].nunique(),\n        },\n        {\n            "check": "hours_per_county",\n            "value": N_HOURS,\n        },\n        {\n            "check": "cutoff_timestamp",\n            "value": str(cutoff_timestamp),\n        },\n        {\n            "check": "future_test_outage_values",\n            "value": 0,\n        },\n        {\n            "check": "weather_variables",\n            "value": len(WEATHER) + 1,\n        },\n    ]\n\n    pd.DataFrame(\n        rows\n    ).to_csv(\n        METRICS_DIR\n        / "input_audit.csv",\n        index=False,\n    )\n\n    print(\n        "PASS: official shapes/counties/times verified; "\n        "no future test outage state is exposed."\n    )\n\n\n# ================================================================\n# 5. PERMANENT COUNTY FOLDS\n# ================================================================\n\ndef create_or_verify_folds(train):\n\n    if FOLD_PATH.exists():\n\n        folds = pd.read_csv(\n            FOLD_PATH\n        )\n\n    else:\n\n        county = (\n            train\n            .groupby(\n                "fipsCode",\n                as_index=False,\n            )\n            .agg(\n                countyName=(\n                    "countyName",\n                    "first",\n                ),\n                stateAbbr=(\n                    "stateAbbr",\n                    "first",\n                ),\n                severity_tier=(\n                    "severity_tier",\n                    "first",\n                ),\n            )\n        )\n\n        county[\n            "strat_label"\n        ] = (\n            county[\n                "stateAbbr"\n            ].astype(str)\n            + "_"\n            + county[\n                "severity_tier"\n            ].astype(str)\n        )\n\n        county["fold"] = -1\n\n        cv = StratifiedGroupKFold(\n            n_splits=N_FOLDS,\n            shuffle=True,\n            random_state=FOLD_SEED,\n        )\n\n        for fold, (_, val_idx) in enumerate(\n            cv.split(\n                county,\n                y=county[\n                    "strat_label"\n                ],\n                groups=county[\n                    "fipsCode"\n                ],\n            )\n        ):\n\n            county.loc[\n                val_idx,\n                "fold",\n            ] = fold\n\n        folds = county\n\n        folds.to_csv(\n            FOLD_PATH,\n            index=False,\n        )\n\n    assert len(folds) == 239\n    assert folds["fipsCode"].nunique() == 239\n    assert set(folds["fold"].unique()) == {0, 1, 2, 3, 4}\n\n    return folds\n\n\n# ================================================================\n# 6. METRICS / COMMON DIAGNOSTICS\n# ================================================================\n\ndef calculate_metrics(\n    actual,\n    predicted,\n):\n\n    actual = np.asarray(\n        actual,\n        dtype=float,\n    )\n\n    predicted = np.asarray(\n        predicted,\n        dtype=float,\n    )\n\n    valid = (\n        np.isfinite(actual)\n        & np.isfinite(predicted)\n    )\n\n    actual = actual[valid]\n    predicted = predicted[valid]\n\n    error = (\n        predicted\n        - actual\n    )\n\n    rmse = float(\n        np.sqrt(\n            np.mean(\n                error ** 2\n            )\n        )\n    )\n\n    mae = float(\n        np.mean(\n            np.abs(\n                error\n            )\n        )\n    )\n\n    if (\n        len(actual) > 2\n        and np.std(actual) > 0\n        and np.std(predicted) > 0\n    ):\n        rho = float(\n            pearsonr(\n                predicted,\n                actual,\n            )[0]\n        )\n    else:\n        rho = np.nan\n\n    denominator = float(\n        np.sum(\n            (\n                actual\n                - actual.mean()\n            ) ** 2\n        )\n    )\n\n    if denominator > 0:\n\n        r2 = float(\n            1.0\n            - np.sum(\n                error ** 2\n            )\n            / denominator\n        )\n\n    else:\n        r2 = np.nan\n\n    return {\n        "n": len(actual),\n        "RMSE": rmse,\n        "MAE": mae,\n        "Pearson_r": rho,\n        "R2": r2,\n    }\n\n\ndef metrics_table(\n    oof,\n    prediction_col="predicted_osi",\n):\n\n    rows = []\n\n    for horizon in HORIZONS:\n\n        subset = oof[\n            oof[\n                "horizon_h"\n            ]\n            == horizon\n        ]\n\n        m = calculate_metrics(\n            subset[\n                "actual_osi"\n            ],\n            subset[\n                prediction_col\n            ],\n        )\n\n        assert (\n            m["n"]\n            == EXPECTED_TARGET_N[\n                horizon\n            ]\n        )\n\n        rows.append(\n            {\n                "horizon":\n                    f"t+{horizon}h",\n                **m,\n            }\n        )\n\n    return pd.DataFrame(\n        rows\n    )\n\n\ndef duplicate_target_discrepancy_long(\n    oof,\n    prediction_col="predicted_osi",\n):\n\n    spread = (\n        oof\n        .groupby(\n            [\n                "fipsCode",\n                "target_timestamp",\n            ]\n        )[\n            prediction_col\n        ]\n        .agg(\n            lambda x:\n                float(\n                    np.max(x)\n                    - np.min(x)\n                )\n        )\n    )\n\n    return float(\n        spread.max()\n    )\n\n\ndef duplicate_target_discrepancy_submission(\n    submission,\n):\n\n    origin = pd.to_datetime(\n        submission[\n            "timestamp_et"\n        ]\n    )\n\n    parts = []\n\n    for horizon, col in zip(\n        HORIZONS,\n        TARGET_COLS,\n    ):\n\n        valid = (\n            submission[col]\n            .notna()\n        )\n\n        parts.append(\n            pd.DataFrame(\n                {\n                    "fipsCode":\n                        submission.loc[\n                            valid,\n                            "fipsCode",\n                        ]\n                        .astype(int)\n                        .to_numpy(),\n\n                    "target_timestamp":\n                        (\n                            origin.loc[\n                                valid\n                            ]\n                            + pd.to_timedelta(\n                                horizon,\n                                unit="h",\n                            )\n                        )\n                        .to_numpy(),\n\n                    "prediction":\n                        submission.loc[\n                            valid,\n                            col,\n                        ]\n                        .to_numpy(float),\n                }\n            )\n        )\n\n    d = pd.concat(\n        parts,\n        ignore_index=True,\n    )\n\n    spread = (\n        d\n        .groupby(\n            [\n                "fipsCode",\n                "target_timestamp",\n            ]\n        )[\n            "prediction"\n        ]\n        .agg(\n            lambda x:\n                float(\n                    np.max(x)\n                    - np.min(x)\n                )\n        )\n    )\n\n    return float(\n        spread.max()\n    )\n\n\n# ================================================================\n# 7. INITIAL / Pias+Munirul-STYLE CLEANED BASELINE (EXP03)\n# ================================================================\n\ndef baseline_feature_lists():\n\n    sequence_cols = (\n        FUTURE_COLS\n        + OUTAGE_STATE\n    )\n\n    return (\n        sequence_cols,\n        FUTURE_COLS.copy(),\n        BASELINE_STATIC.copy(),\n    )\n\n\ndef freeze_baseline_outage_state(df):\n\n    result = df.copy()\n\n    for _, indexes in (\n        result\n        .groupby("fipsCode")\n        .indices\n        .items()\n    ):\n\n        indexes = np.asarray(\n            indexes\n        )\n\n        anchor = (\n            result.iloc[\n                indexes[\n                    CUTOFF - 1\n                ]\n            ][\n                OUTAGE_STATE\n            ]\n            .to_numpy(float)\n        )\n\n        if not np.isfinite(\n            anchor\n        ).all():\n\n            raise ValueError(\n                "Missing cutoff outage state."\n            )\n\n        result.loc[\n            indexes[\n                CUTOFF:\n            ],\n            OUTAGE_STATE,\n        ] = anchor\n\n    return result\n\n\ndef baseline_scale(\n    frozen,\n    train_ids,\n    sequence_cols,\n):\n\n    scaler = StandardScaler()\n\n    fit_array = (\n        frozen.loc[\n            frozen[\n                "fipsCode"\n            ].isin(\n                train_ids\n            ),\n            sequence_cols,\n        ]\n        .astype(float)\n        .to_numpy()\n    )\n\n    scaler.fit(\n        fit_array\n    )\n\n    sequence_scaled = scaler.transform(\n        frozen[\n            sequence_cols\n        ]\n        .astype(float)\n        .to_numpy()\n    ).astype(\n        np.float32\n    )\n\n    return (\n        sequence_scaled,\n        scaler,\n    )\n\n\ndef baseline_build_windows(\n    frozen,\n    original,\n    county_ids,\n    sequence_cols,\n    future_cols,\n    static_cols,\n    scaler,\n):\n\n    sequence = scaler.transform(\n        frozen[\n            sequence_cols\n        ]\n        .astype(float)\n        .to_numpy()\n    ).astype(\n        np.float32\n    )\n\n    # Future weather columns occupy the same values/scales as the\n    # corresponding columns in the sequence scaler. Extract their\n    # positions from sequence_cols.\n    future_positions_in_sequence = [\n        sequence_cols.index(\n            col\n        )\n        for col in future_cols\n    ]\n\n    future = sequence[\n        :,\n        future_positions_in_sequence,\n    ]\n\n    static = (\n        frozen[\n            static_cols\n        ]\n        .to_numpy(\n            dtype=np.float32\n        )\n    )\n\n    groups = {\n        int(fips):\n            np.asarray(\n                indexes\n            )\n        for fips, indexes\n        in original\n        .groupby(\n            "fipsCode"\n        )\n        .indices\n        .items()\n    }\n\n    xs = []\n    xf = []\n    xstatic = []\n    target_model = []\n    target_raw = []\n    masks = []\n    keys = []\n\n    for county in county_ids:\n\n        county = int(county)\n\n        indexes = groups[\n            county\n        ]\n\n        assert len(\n            indexes\n        ) == N_HOURS\n\n        for t in range(\n            CUTOFF,\n            N_HOURS,\n        ):\n\n            history_idx = indexes[\n                t\n                - BASELINE_WINDOW\n                + 1:\n                t\n                + 1\n            ]\n\n            target_weather_idx = [\n                indexes[\n                    min(\n                        t + horizon,\n                        N_HOURS - 1,\n                    )\n                ]\n                for horizon\n                in HORIZONS\n            ]\n\n            raw_y = (\n                original.iloc[\n                    indexes[t]\n                ][\n                    TARGET_COLS\n                ]\n                .to_numpy(\n                    dtype=float\n                )\n            )\n\n            valid = np.isfinite(\n                raw_y\n            )\n\n            y_model = np.zeros(\n                len(\n                    HORIZONS\n                ),\n                dtype=np.float32,\n            )\n\n            y_model[\n                valid\n            ] = np.log1p(\n                np.maximum(\n                    raw_y[\n                        valid\n                    ],\n                    0.0,\n                )\n            ).astype(\n                np.float32\n            )\n\n            xs.append(\n                sequence[\n                    history_idx\n                ]\n            )\n\n            xf.append(\n                future[\n                    target_weather_idx\n                ]\n            )\n\n            xstatic.append(\n                static[\n                    indexes[t]\n                ]\n            )\n\n            target_model.append(\n                y_model\n            )\n\n            target_raw.append(\n                raw_y\n            )\n\n            masks.append(\n                valid.astype(\n                    np.float32\n                )\n            )\n\n            origin_time = pd.Timestamp(\n                original.iloc[\n                    indexes[t]\n                ][\n                    "timestamp_et"\n                ]\n            )\n\n            keys.append(\n                (\n                    county,\n                    origin_time,\n                    t,\n                )\n            )\n\n    arrays = (\n        np.asarray(\n            xs,\n            dtype=np.float32,\n        ),\n        np.asarray(\n            xf,\n            dtype=np.float32,\n        ),\n        np.asarray(\n            xstatic,\n            dtype=np.float32,\n        ),\n        np.asarray(\n            target_model,\n            dtype=np.float32,\n        ),\n        np.asarray(\n            masks,\n            dtype=np.float32,\n        ),\n        np.asarray(\n            target_raw,\n            dtype=np.float64,\n        ),\n    )\n\n    key_df = pd.DataFrame(\n        keys,\n        columns=[\n            "fipsCode",\n            "origin_timestamp",\n            "origin_tau",\n        ],\n    )\n\n    return (\n        arrays,\n        key_df,\n    )\n\n\nclass BaselineCNNLSTM(nn.Module):\n\n    def __init__(\n        self,\n        n_sequence,\n        n_future_flat,\n        n_static,\n        n_outputs,\n        window,\n    ):\n\n        super().__init__()\n\n        self.cnn = nn.Sequential(\n            nn.Conv1d(\n                n_sequence,\n                32,\n                kernel_size=2,\n                padding="same",\n            ),\n            nn.ReLU(),\n            nn.Conv1d(\n                32,\n                32,\n                kernel_size=2,\n                padding="same",\n            ),\n            nn.ReLU(),\n            nn.MaxPool1d(2),\n            nn.Dropout(0.1),\n            nn.Flatten(),\n        )\n\n        self.lstm = nn.LSTM(\n            n_sequence,\n            50,\n            batch_first=True,\n        )\n\n        self.future = nn.Sequential(\n            nn.Flatten(),\n            nn.Linear(\n                n_future_flat,\n                32,\n            ),\n            nn.ReLU(),\n        )\n\n        head_input = (\n            32\n            * (\n                window // 2\n            )\n            + 50\n            + 32\n            + n_static\n        )\n\n        self.head = nn.Sequential(\n            nn.Linear(\n                head_input,\n                64,\n            ),\n            nn.ReLU(),\n            nn.Dropout(0.2),\n            nn.Linear(\n                64,\n                n_outputs,\n            ),\n        )\n\n    def forward(\n        self,\n        sequence,\n        future,\n        static,\n    ):\n\n        cnn = self.cnn(\n            sequence.transpose(\n                1,\n                2,\n            )\n        )\n\n        lstm_output, _ = self.lstm(\n            sequence\n        )\n\n        lstm = torch.relu(\n            lstm_output[\n                :,\n                -1,\n                :\n            ]\n        )\n\n        future = self.future(\n            future\n        )\n\n        return self.head(\n            torch.cat(\n                [\n                    cnn,\n                    lstm,\n                    future,\n                    static,\n                ],\n                dim=1,\n            )\n        )\n\n\ndef masked_smooth_l1(\n    prediction,\n    target,\n    mask,\n):\n\n    element = (\n        nn.functional\n        .smooth_l1_loss(\n            prediction,\n            target,\n            reduction="none",\n        )\n    )\n\n    return (\n        (\n            element\n            * mask\n        ).sum()\n        / mask.sum()\n        .clamp(\n            min=1.0\n        )\n    )\n\n\ndef baseline_train_model(\n    train_arrays,\n    seed,\n):\n\n    set_seed(\n        seed\n    )\n\n    (\n        xs,\n        xf,\n        xstatic,\n        target,\n        mask,\n        _,\n    ) = train_arrays\n\n    dataset = (\n        torch.utils.data\n        .TensorDataset(\n            torch.from_numpy(\n                xs\n            ),\n            torch.from_numpy(\n                xf\n            ),\n            torch.from_numpy(\n                xstatic\n            ),\n            torch.from_numpy(\n                target\n            ),\n            torch.from_numpy(\n                mask\n            ),\n        )\n    )\n\n    generator = (\n        torch.Generator()\n    )\n\n    generator.manual_seed(\n        seed\n    )\n\n    loader = DataLoader(\n        dataset,\n        batch_size=\n            BASELINE_BATCH_SIZE,\n        shuffle=True,\n        num_workers=0,\n        pin_memory=(\n            DEVICE.type\n            == "cuda"\n        ),\n        generator=generator,\n    )\n\n    sequence_cols, future_cols, static_cols = (\n        baseline_feature_lists()\n    )\n\n    model = BaselineCNNLSTM(\n        n_sequence=len(\n            sequence_cols\n        ),\n        n_future_flat=(\n            len(\n                future_cols\n            )\n            * len(\n                HORIZONS\n            )\n        ),\n        n_static=len(\n            static_cols\n        ),\n        n_outputs=len(\n            HORIZONS\n        ),\n        window=BASELINE_WINDOW,\n    ).to(\n        DEVICE\n    )\n\n    optimizer = (\n        torch.optim.Adam(\n            model.parameters(),\n            lr=BASELINE_LR,\n        )\n    )\n\n    history = []\n\n    for epoch in range(\n        1,\n        BASELINE_EPOCHS\n        + 1,\n    ):\n\n        model.train()\n\n        running = 0.0\n        weight = 0.0\n\n        for (\n            sequence,\n            future,\n            static,\n            target_batch,\n            mask_batch,\n        ) in loader:\n\n            sequence = sequence.to(\n                DEVICE,\n                non_blocking=True,\n            )\n\n            future = future.to(\n                DEVICE,\n                non_blocking=True,\n            )\n\n            static = static.to(\n                DEVICE,\n                non_blocking=True,\n            )\n\n            target_batch = (\n                target_batch.to(\n                    DEVICE,\n                    non_blocking=True,\n                )\n            )\n\n            mask_batch = (\n                mask_batch.to(\n                    DEVICE,\n                    non_blocking=True,\n                )\n            )\n\n            optimizer.zero_grad(\n                set_to_none=True\n            )\n\n            prediction = model(\n                sequence,\n                future,\n                static,\n            )\n\n            loss = masked_smooth_l1(\n                prediction,\n                target_batch,\n                mask_batch,\n            )\n\n            loss.backward()\n            optimizer.step()\n\n            n_valid = float(\n                mask_batch\n                .sum()\n                .item()\n            )\n\n            running += (\n                float(\n                    loss.item()\n                )\n                * n_valid\n            )\n\n            weight += n_valid\n\n        history.append(\n            {\n                "epoch":\n                    epoch,\n\n                "train_loss":\n                    running\n                    / max(\n                        weight,\n                        1.0,\n                    ),\n            }\n        )\n\n    return (\n        model,\n        pd.DataFrame(\n            history\n        ),\n    )\n\n\ndef baseline_predict(\n    model,\n    arrays,\n):\n\n    (\n        xs,\n        xf,\n        xstatic,\n        _,\n        _,\n        _,\n    ) = arrays\n\n    dataset = (\n        torch.utils.data\n        .TensorDataset(\n            torch.from_numpy(\n                xs\n            ),\n            torch.from_numpy(\n                xf\n            ),\n            torch.from_numpy(\n                xstatic\n            ),\n        )\n    )\n\n    loader = DataLoader(\n        dataset,\n        batch_size=\n            BASELINE_BATCH_SIZE,\n        shuffle=False,\n        num_workers=0,\n        pin_memory=(\n            DEVICE.type\n            == "cuda"\n        ),\n    )\n\n    model.eval()\n\n    outputs = []\n\n    with torch.no_grad():\n\n        for (\n            sequence,\n            future,\n            static,\n        ) in loader:\n\n            prediction = model(\n                sequence.to(\n                    DEVICE,\n                    non_blocking=True,\n                ),\n                future.to(\n                    DEVICE,\n                    non_blocking=True,\n                ),\n                static.to(\n                    DEVICE,\n                    non_blocking=True,\n                ),\n            )\n\n            outputs.append(\n                prediction\n                .cpu()\n                .numpy()\n            )\n\n    prediction = np.concatenate(\n        outputs,\n        axis=0,\n    )\n\n    prediction = np.expm1(\n        prediction\n    )\n\n    return np.clip(\n        prediction,\n        0.0,\n        None,\n    )\n\n\ndef baseline_long_oof(\n    keys,\n    arrays,\n    prediction,\n    fold,\n):\n\n    raw_target = arrays[\n        5\n    ]\n\n    mask = arrays[\n        4\n    ]\n\n    parts = []\n\n    for j, horizon in enumerate(\n        HORIZONS\n    ):\n\n        valid = (\n            mask[\n                :,\n                j,\n            ]\n            > 0\n        )\n\n        part = (\n            keys.loc[\n                valid\n            ]\n            .copy()\n        )\n\n        part[\n            "fold"\n        ] = fold\n\n        part[\n            "horizon_h"\n        ] = horizon\n\n        part[\n            "target_timestamp"\n        ] = (\n            part[\n                "origin_timestamp"\n            ]\n            + pd.to_timedelta(\n                horizon,\n                unit="h",\n            )\n        )\n\n        part[\n            "actual_osi"\n        ] = (\n            raw_target[\n                valid,\n                j,\n            ]\n        )\n\n        part[\n            "predicted_osi"\n        ] = (\n            prediction[\n                valid,\n                j,\n            ]\n        )\n\n        parts.append(\n            part\n        )\n\n    return pd.concat(\n        parts,\n        ignore_index=True,\n    )\n\n\ndef run_baseline_cv(\n    train,\n    folds,\n    force=False,\n):\n\n    oof_path = (\n        OOF_DIR\n        / "BASELINE_EXP03_cleaned_oof.csv"\n    )\n\n    metrics_path = (\n        METRICS_DIR\n        / "BASELINE_EXP03_cleaned_metrics.csv"\n    )\n\n    history_path = (\n        METRICS_DIR\n        / "BASELINE_EXP03_training_history.csv"\n    )\n\n    if (\n        oof_path.exists()\n        and metrics_path.exists()\n        and history_path.exists()\n        and not force\n    ):\n\n        print(\n            "\\nReusing cached cleaned baseline CV."\n        )\n\n        return (\n            pd.read_csv(\n                oof_path,\n                parse_dates=[\n                    "origin_timestamp",\n                    "target_timestamp",\n                ],\n            ),\n            pd.read_csv(\n                metrics_path\n            ),\n        )\n\n    print(\n        "\\n"\n        + "=" * 78\n    )\n\n    print(\n        "BASELINE CV: CLEANED ORIGINAL CNN-LSTM (EXP03)"\n    )\n\n    print(\n        "=" * 78\n    )\n\n    frozen = (\n        freeze_baseline_outage_state(\n            train\n        )\n    )\n\n    sequence_cols, future_cols, static_cols = (\n        baseline_feature_lists()\n    )\n\n    all_oof = []\n    all_history = []\n\n    for fold in range(\n        N_FOLDS\n    ):\n\n        started = time.time()\n\n        train_ids = (\n            folds.loc[\n                folds[\n                    "fold"\n                ]\n                != fold,\n                "fipsCode",\n            ]\n            .astype(int)\n            .to_numpy()\n        )\n\n        val_ids = (\n            folds.loc[\n                folds[\n                    "fold"\n                ]\n                == fold,\n                "fipsCode",\n            ]\n            .astype(int)\n            .to_numpy()\n        )\n\n        _, scaler = baseline_scale(\n            frozen,\n            train_ids,\n            sequence_cols,\n        )\n\n        train_arrays, _ = (\n            baseline_build_windows(\n                frozen,\n                train,\n                train_ids,\n                sequence_cols,\n                future_cols,\n                static_cols,\n                scaler,\n            )\n        )\n\n        val_arrays, keys = (\n            baseline_build_windows(\n                frozen,\n                train,\n                val_ids,\n                sequence_cols,\n                future_cols,\n                static_cols,\n                scaler,\n            )\n        )\n\n        model, history = (\n            baseline_train_model(\n                train_arrays,\n                BASELINE_SEED\n                + fold,\n            )\n        )\n\n        prediction = (\n            baseline_predict(\n                model,\n                val_arrays,\n            )\n        )\n\n        fold_oof = (\n            baseline_long_oof(\n                keys,\n                val_arrays,\n                prediction,\n                fold,\n            )\n        )\n\n        all_oof.append(\n            fold_oof\n        )\n\n        history.insert(\n            0,\n            "fold",\n            fold,\n        )\n\n        all_history.append(\n            history\n        )\n\n        print(\n            f"Baseline fold {fold} complete "\n            f"in {time.time()-started:.1f}s"\n        )\n\n        del model\n\n        if torch.cuda.is_available():\n            torch.cuda.empty_cache()\n\n    oof = pd.concat(\n        all_oof,\n        ignore_index=True,\n    )\n\n    oof = (\n        oof\n        .sort_values(\n            [\n                "fipsCode",\n                "origin_timestamp",\n                "horizon_h",\n            ]\n        )\n        .reset_index(\n            drop=True\n        )\n    )\n\n    metrics = metrics_table(\n        oof\n    )\n\n    oof.to_csv(\n        oof_path,\n        index=False,\n    )\n\n    metrics.to_csv(\n        metrics_path,\n        index=False,\n    )\n\n    pd.concat(\n        all_history,\n        ignore_index=True,\n    ).to_csv(\n        history_path,\n        index=False,\n    )\n\n    discrepancy = (\n        duplicate_target_discrepancy_long(\n            oof\n        )\n    )\n\n    pd.DataFrame(\n        [\n            {\n                "model":\n                    "cleaned_initial_CNN_LSTM",\n\n                "max_duplicate_target_discrepancy":\n                    discrepancy,\n            }\n        ]\n    ).to_csv(\n        METRICS_DIR\n        / "BASELINE_EXP03_target_consistency.csv",\n        index=False,\n    )\n\n    print(\n        "\\nCleaned baseline pooled metrics:"\n    )\n\n    print(\n        metrics\n        .round(6)\n        .to_string(\n            index=False\n        )\n    )\n\n    return (\n        oof,\n        metrics,\n    )\n\n\n# ================================================================\n# 8. FINAL EXP05 MODEL\n# ================================================================\n\ndef fit_final_scalers(\n    df,\n    train_ids,\n):\n\n    county_mask = (\n        df[\n            "fipsCode"\n        ]\n        .isin(\n            train_ids\n        )\n    )\n\n    history_mask = (\n        county_mask\n        & (\n            df[\n                "hour_idx"\n            ]\n            < CUTOFF\n        )\n    )\n\n    future_mask = (\n        county_mask\n        & (\n            df[\n                "hour_idx"\n            ]\n            >= CUTOFF\n        )\n    )\n\n    history_scaler = (\n        StandardScaler()\n    )\n\n    history_scaler.fit(\n        df.loc[\n            history_mask,\n            HISTORY_COLS,\n        ]\n        .astype(float)\n        .to_numpy()\n    )\n\n    future_scaler = (\n        StandardScaler()\n    )\n\n    future_scaler.fit(\n        df.loc[\n            future_mask,\n            FUTURE_COLS,\n        ]\n        .astype(float)\n        .to_numpy()\n    )\n\n    return (\n        history_scaler,\n        future_scaler,\n    )\n\n\ndef build_final_county_arrays(\n    df,\n    county_ids,\n    history_scaler,\n    future_scaler,\n    with_targets,\n):\n\n    histories = []\n    futures = []\n    timestamps = []\n    county_order = []\n\n    target_raw = []\n    target_model = []\n    masks = []\n\n    for fips in county_ids:\n\n        fips = int(\n            fips\n        )\n\n        d = (\n            df[\n                df[\n                    "fipsCode"\n                ]\n                == fips\n            ]\n            .sort_values(\n                "timestamp_et"\n            )\n            .reset_index(\n                drop=True\n            )\n        )\n\n        assert len(\n            d\n        ) == N_HOURS\n\n        history_raw = (\n            d.iloc[\n                :CUTOFF\n            ][\n                HISTORY_COLS\n            ]\n            .astype(float)\n            .to_numpy()\n        )\n\n        if not np.isfinite(\n            history_raw\n        ).all():\n\n            raise ValueError(\n                f"Non-finite observed history for county {fips}."\n            )\n\n        future_raw = (\n            d.iloc[\n                CUTOFF:\n            ][\n                FUTURE_COLS\n            ]\n            .astype(float)\n            .to_numpy()\n        )\n\n        if not np.isfinite(\n            future_raw\n        ).all():\n\n            raise ValueError(\n                f"Non-finite future weather for county {fips}."\n            )\n\n        histories.append(\n            history_scaler\n            .transform(\n                history_raw\n            )\n            .astype(\n                np.float32\n            )\n        )\n\n        futures.append(\n            future_scaler\n            .transform(\n                future_raw\n            )\n            .astype(\n                np.float32\n            )\n        )\n\n        timestamps.append(\n            d.iloc[\n                CUTOFF:\n            ][\n                "timestamp_et"\n            ]\n            .to_numpy()\n        )\n\n        county_order.append(\n            fips\n        )\n\n        if with_targets:\n\n            raw = (\n                d.iloc[\n                    CUTOFF:\n                ][\n                    TARGET_COLS\n                ]\n                .to_numpy(\n                    dtype=np.float64\n                )\n            )\n\n            valid = np.isfinite(\n                raw\n            )\n\n            model_target = np.zeros(\n                raw.shape,\n                dtype=np.float32,\n            )\n\n            model_target[\n                valid\n            ] = np.log1p(\n                np.maximum(\n                    raw[\n                        valid\n                    ],\n                    0.0,\n                )\n            ).astype(\n                np.float32\n            )\n\n            target_raw.append(\n                raw\n            )\n\n            target_model.append(\n                model_target\n            )\n\n            masks.append(\n                valid.astype(\n                    np.float32\n                )\n            )\n\n    result = {\n        "history":\n            np.stack(\n                histories\n            ).astype(\n                np.float32\n            ),\n\n        "future":\n            np.stack(\n                futures\n            ).astype(\n                np.float32\n            ),\n\n        "timestamps":\n            np.stack(\n                timestamps\n            ),\n\n        "county_ids":\n            np.asarray(\n                county_order,\n                dtype=np.int64,\n            ),\n    }\n\n    if with_targets:\n\n        result[\n            "target_raw"\n        ] = np.stack(\n            target_raw\n        )\n\n        result[\n            "target_model"\n        ] = np.stack(\n            target_model\n        )\n\n        result[\n            "mask"\n        ] = np.stack(\n            masks\n        )\n\n    return result\n\n\nclass FinalCountyDataset(Dataset):\n\n    def __init__(\n        self,\n        arrays,\n        with_targets,\n    ):\n\n        self.history = arrays[\n            "history"\n        ]\n\n        self.future = arrays[\n            "future"\n        ]\n\n        self.with_targets = (\n            with_targets\n        )\n\n        if with_targets:\n\n            self.target = arrays[\n                "target_model"\n            ]\n\n            self.mask = arrays[\n                "mask"\n            ]\n\n    def __len__(self):\n\n        return len(\n            self.history\n        )\n\n    def __getitem__(\n        self,\n        index,\n    ):\n\n        history = torch.from_numpy(\n            self.history[\n                index\n            ]\n        )\n\n        future = torch.from_numpy(\n            self.future[\n                index\n            ]\n        )\n\n        if not self.with_targets:\n\n            return (\n                history,\n                future,\n            )\n\n        return (\n            history,\n            future,\n            torch.from_numpy(\n                self.target[\n                    index\n                ]\n            ),\n            torch.from_numpy(\n                self.mask[\n                    index\n                ]\n            ),\n        )\n\n\nclass FinalCNNV2(nn.Module):\n\n    def __init__(self):\n\n        super().__init__()\n\n        self.history_cnn = (\n            nn.Sequential(\n                nn.Conv1d(\n                    len(\n                        HISTORY_COLS\n                    ),\n                    32,\n                    kernel_size=3,\n                    padding=1,\n                ),\n                nn.ReLU(),\n                nn.Conv1d(\n                    32,\n                    32,\n                    kernel_size=3,\n                    padding=1,\n                ),\n                nn.ReLU(),\n            )\n        )\n\n        self.history_lstm = (\n            nn.LSTM(\n                input_size=32,\n                hidden_size=64,\n                batch_first=True,\n            )\n        )\n\n        self.future_lstm = (\n            nn.LSTM(\n                input_size=len(\n                    FUTURE_COLS\n                ),\n                hidden_size=64,\n                batch_first=True,\n            )\n        )\n\n        self.decoder = (\n            nn.Sequential(\n                nn.Linear(\n                    129,\n                    64,\n                ),\n                nn.ReLU(),\n                nn.Dropout(\n                    0.10\n                ),\n                nn.Linear(\n                    64,\n                    1,\n                ),\n            )\n        )\n\n        target_index = (\n            np.arange(\n                N_FUTURE\n            )[\n                :,\n                None,\n            ]\n            + np.asarray(\n                HORIZONS\n            )[\n                None,\n                :,\n            ]\n        )\n\n        target_index = np.clip(\n            target_index,\n            0,\n            N_FUTURE - 1,\n        )\n\n        target_lead = (\n            target_index\n            / float(\n                N_FUTURE - 1\n            )\n        )\n\n        self.register_buffer(\n            "target_index",\n            torch.tensor(\n                target_index,\n                dtype=torch.long,\n            ),\n        )\n\n        self.register_buffer(\n            "target_lead",\n            torch.tensor(\n                target_lead,\n                dtype=torch.float32,\n            ).unsqueeze(\n                -1\n            ),\n        )\n\n    def forward(\n        self,\n        history,\n        future,\n    ):\n\n        batch = (\n            history.shape[\n                0\n            ]\n        )\n\n        h = history.transpose(\n            1,\n            2,\n        )\n\n        h = self.history_cnn(\n            h\n        )\n\n        h = h.transpose(\n            1,\n            2,\n        )\n\n        history_out, _ = (\n            self.history_lstm(\n                h\n            )\n        )\n\n        history_embedding = (\n            history_out[\n                :,\n                -1,\n                :\n            ]\n        )\n\n        future_out, _ = (\n            self.future_lstm(\n                future\n            )\n        )\n\n        indices = (\n            self.target_index\n            .unsqueeze(\n                0\n            )\n            .expand(\n                batch,\n                -1,\n                -1,\n            )\n        )\n\n        flat = indices.reshape(\n            batch,\n            -1,\n        )\n\n        gather_index = (\n            flat\n            .unsqueeze(\n                -1\n            )\n            .expand(\n                -1,\n                -1,\n                future_out.shape[\n                    -1\n                ],\n            )\n        )\n\n        weather_target = (\n            torch.gather(\n                future_out,\n                dim=1,\n                index=gather_index,\n            )\n            .reshape(\n                batch,\n                N_FUTURE,\n                len(\n                    HORIZONS\n                ),\n                64,\n            )\n        )\n\n        history_target = (\n            history_embedding[\n                :,\n                None,\n                None,\n                :\n            ]\n            .expand(\n                -1,\n                N_FUTURE,\n                len(\n                    HORIZONS\n                ),\n                -1,\n            )\n        )\n\n        lead = (\n            self.target_lead\n            .unsqueeze(\n                0\n            )\n            .expand(\n                batch,\n                -1,\n                -1,\n                -1,\n            )\n        )\n\n        decoder_input = (\n            torch.cat(\n                [\n                    history_target,\n                    weather_target,\n                    lead,\n                ],\n                dim=-1,\n            )\n        )\n\n        return (\n            self.decoder(\n                decoder_input\n            )\n            .squeeze(\n                -1\n            )\n        )\n\n\ndef final_train_model(\n    arrays,\n    seed,\n):\n\n    set_seed(\n        seed\n    )\n\n    dataset = FinalCountyDataset(\n        arrays,\n        with_targets=True,\n    )\n\n    generator = (\n        torch.Generator()\n    )\n\n    generator.manual_seed(\n        seed\n    )\n\n    loader = DataLoader(\n        dataset,\n        batch_size=\n            FINAL_BATCH_SIZE,\n        shuffle=True,\n        num_workers=0,\n        pin_memory=(\n            DEVICE.type\n            == "cuda"\n        ),\n        generator=generator,\n    )\n\n    model = FinalCNNV2().to(\n        DEVICE\n    )\n\n    optimizer = (\n        torch.optim.Adam(\n            model.parameters(),\n            lr=FINAL_LR,\n        )\n    )\n\n    history_rows = []\n\n    for epoch in range(\n        1,\n        FINAL_EPOCHS\n        + 1,\n    ):\n\n        model.train()\n\n        running = 0.0\n        total_valid = 0.0\n\n        for (\n            history,\n            future,\n            target,\n            mask,\n        ) in loader:\n\n            history = history.to(\n                DEVICE,\n                non_blocking=True,\n            )\n\n            future = future.to(\n                DEVICE,\n                non_blocking=True,\n            )\n\n            target = target.to(\n                DEVICE,\n                non_blocking=True,\n            )\n\n            mask = mask.to(\n                DEVICE,\n                non_blocking=True,\n            )\n\n            optimizer.zero_grad(\n                set_to_none=True\n            )\n\n            prediction = model(\n                history,\n                future,\n            )\n\n            loss = masked_smooth_l1(\n                prediction,\n                target,\n                mask,\n            )\n\n            loss.backward()\n            optimizer.step()\n\n            n_valid = float(\n                mask\n                .sum()\n                .item()\n            )\n\n            running += (\n                float(\n                    loss.item()\n                )\n                * n_valid\n            )\n\n            total_valid += (\n                n_valid\n            )\n\n        epoch_loss = (\n            running\n            / max(\n                total_valid,\n                1.0,\n            )\n        )\n\n        history_rows.append(\n            {\n                "epoch":\n                    epoch,\n\n                "train_loss":\n                    epoch_loss,\n            }\n        )\n\n        if (\n            epoch == 1\n            or epoch % 20 == 0\n            or epoch == FINAL_EPOCHS\n        ):\n\n            print(\n                f"      epoch {epoch:3d}/"\n                f"{FINAL_EPOCHS}: "\n                f"loss={epoch_loss:.8f}"\n            )\n\n    return (\n        model,\n        pd.DataFrame(\n            history_rows\n        ),\n    )\n\n\ndef final_predict(\n    model,\n    arrays,\n):\n\n    dataset = FinalCountyDataset(\n        arrays,\n        with_targets=False,\n    )\n\n    loader = DataLoader(\n        dataset,\n        batch_size=\n            FINAL_BATCH_SIZE,\n        shuffle=False,\n        num_workers=0,\n        pin_memory=(\n            DEVICE.type\n            == "cuda"\n        ),\n    )\n\n    model.eval()\n\n    outputs = []\n\n    with torch.no_grad():\n\n        for (\n            history,\n            future,\n        ) in loader:\n\n            prediction = model(\n                history.to(\n                    DEVICE,\n                    non_blocking=True,\n                ),\n                future.to(\n                    DEVICE,\n                    non_blocking=True,\n                ),\n            )\n\n            outputs.append(\n                prediction\n                .cpu()\n                .numpy()\n            )\n\n    prediction_log = np.concatenate(\n        outputs,\n        axis=0,\n    )\n\n    return (\n        np.clip(\n            np.expm1(\n                prediction_log\n            ),\n            0.0,\n            None,\n        )\n        .astype(\n            np.float32\n        )\n    )\n\n\ndef final_long_oof(\n    arrays,\n    prediction,\n    fold,\n):\n\n    parts = []\n\n    for county_i, fips in enumerate(\n        arrays[\n            "county_ids"\n        ]\n    ):\n\n        for j, horizon in enumerate(\n            HORIZONS\n        ):\n\n            valid = (\n                arrays[\n                    "mask"\n                ][\n                    county_i,\n                    :,\n                    j,\n                ]\n                > 0\n            )\n\n            origin_times = pd.to_datetime(\n                arrays[\n                    "timestamps"\n                ][\n                    county_i,\n                    valid,\n                ]\n            )\n\n            part = pd.DataFrame(\n                {\n                    "fipsCode":\n                        int(\n                            fips\n                        ),\n\n                    "origin_timestamp":\n                        origin_times,\n\n                    "horizon_h":\n                        horizon,\n\n                    "target_timestamp":\n                        (\n                            origin_times\n                            + pd.to_timedelta(\n                                horizon,\n                                unit="h",\n                            )\n                        ),\n\n                    "fold":\n                        fold,\n\n                    "actual_osi":\n                        arrays[\n                            "target_raw"\n                        ][\n                            county_i,\n                            valid,\n                            j,\n                        ],\n\n                    "predicted_osi":\n                        prediction[\n                            county_i,\n                            valid,\n                            j,\n                        ],\n                }\n            )\n\n            parts.append(\n                part\n            )\n\n    return pd.concat(\n        parts,\n        ignore_index=True,\n    )\n\n\ndef run_final_cv(\n    train,\n    folds,\n    force=False,\n):\n\n    seed_oofs = {}\n\n    seed_metrics_rows = []\n\n    for seed in FINAL_SEEDS:\n\n        oof_path = (\n            OOF_DIR\n            / f"FINAL_EXP05_seed{seed}_oof.csv"\n        )\n\n        history_path = (\n            METRICS_DIR\n            / f"FINAL_EXP05_seed{seed}_training_history.csv"\n        )\n\n        if (\n            oof_path.exists()\n            and history_path.exists()\n            and not force\n        ):\n\n            print(\n                f"\\nReusing final CV seed {seed}."\n            )\n\n            oof = pd.read_csv(\n                oof_path,\n                parse_dates=[\n                    "origin_timestamp",\n                    "target_timestamp",\n                ],\n            )\n\n        else:\n\n            print(\n                "\\n"\n                + "=" * 78\n            )\n\n            print(\n                f"FINAL EXP05 CV: seed {seed}"\n            )\n\n            print(\n                "=" * 78\n            )\n\n            fold_parts = []\n            histories = []\n\n            for fold in range(\n                N_FOLDS\n            ):\n\n                started = time.time()\n\n                train_ids = (\n                    folds.loc[\n                        folds[\n                            "fold"\n                        ]\n                        != fold,\n                        "fipsCode",\n                    ]\n                    .astype(int)\n                    .to_numpy()\n                )\n\n                val_ids = (\n                    folds.loc[\n                        folds[\n                            "fold"\n                        ]\n                        == fold,\n                        "fipsCode",\n                    ]\n                    .astype(int)\n                    .to_numpy()\n                )\n\n                (\n                    history_scaler,\n                    future_scaler,\n                ) = fit_final_scalers(\n                    train,\n                    train_ids,\n                )\n\n                train_arrays = (\n                    build_final_county_arrays(\n                        train,\n                        train_ids,\n                        history_scaler,\n                        future_scaler,\n                        with_targets=True,\n                    )\n                )\n\n                val_arrays = (\n                    build_final_county_arrays(\n                        train,\n                        val_ids,\n                        history_scaler,\n                        future_scaler,\n                        with_targets=True,\n                    )\n                )\n\n                model, history = (\n                    final_train_model(\n                        train_arrays,\n                        seed\n                        + fold,\n                    )\n                )\n\n                prediction = (\n                    final_predict(\n                        model,\n                        val_arrays,\n                    )\n                )\n\n                fold_oof = (\n                    final_long_oof(\n                        val_arrays,\n                        prediction,\n                        fold,\n                    )\n                )\n\n                fold_parts.append(\n                    fold_oof\n                )\n\n                history.insert(\n                    0,\n                    "fold",\n                    fold,\n                )\n\n                history.insert(\n                    0,\n                    "seed",\n                    seed,\n                )\n\n                histories.append(\n                    history\n                )\n\n                print(\n                    f"  seed {seed}, fold {fold}: "\n                    f"{time.time()-started:.1f}s"\n                )\n\n                del model\n\n                if torch.cuda.is_available():\n                    torch.cuda.empty_cache()\n\n            oof = pd.concat(\n                fold_parts,\n                ignore_index=True,\n            )\n\n            oof = (\n                oof\n                .sort_values(\n                    [\n                        "fipsCode",\n                        "origin_timestamp",\n                        "horizon_h",\n                    ]\n                )\n                .reset_index(\n                    drop=True\n                )\n            )\n\n            oof.to_csv(\n                oof_path,\n                index=False,\n            )\n\n            pd.concat(\n                histories,\n                ignore_index=True,\n            ).to_csv(\n                history_path,\n                index=False,\n            )\n\n        seed_oofs[\n            seed\n        ] = oof\n\n        m = metrics_table(\n            oof\n        )\n\n        for row in (\n            m\n            .itertuples(\n                index=False\n            )\n        ):\n\n            seed_metrics_rows.append(\n                {\n                    "seed":\n                        seed,\n\n                    "horizon":\n                        row.horizon,\n\n                    "RMSE":\n                        row.RMSE,\n\n                    "MAE":\n                        row.MAE,\n\n                    "Pearson_r":\n                        row.Pearson_r,\n\n                    "R2":\n                        row.R2,\n                }\n            )\n\n    seed_metrics = pd.DataFrame(\n        seed_metrics_rows\n    )\n\n    seed_metrics.to_csv(\n        METRICS_DIR\n        / "FINAL_EXP05_per_seed_metrics.csv",\n        index=False,\n    )\n\n    seed_summary = (\n        seed_metrics\n        .groupby(\n            "horizon",\n            as_index=False,\n        )\n        .agg(\n            mean_RMSE=(\n                "RMSE",\n                "mean",\n            ),\n            std_RMSE=(\n                "RMSE",\n                "std",\n            ),\n            min_RMSE=(\n                "RMSE",\n                "min",\n            ),\n            max_RMSE=(\n                "RMSE",\n                "max",\n            ),\n        )\n    )\n\n    seed_summary.to_csv(\n        METRICS_DIR\n        / "FINAL_EXP05_seed_summary.csv",\n        index=False,\n    )\n\n    final_oof = (\n        average_oof_predictions(\n            seed_oofs\n        )\n    )\n\n    final_metrics = (\n        metrics_table(\n            final_oof\n        )\n    )\n\n    final_oof.to_csv(\n        OOF_DIR\n        / "FINAL_EXP05_5SEED_AVERAGE_oof.csv",\n        index=False,\n    )\n\n    final_metrics.to_csv(\n        METRICS_DIR\n        / "FINAL_EXP05_5SEED_AVERAGE_metrics.csv",\n        index=False,\n    )\n\n    discrepancy = (\n        duplicate_target_discrepancy_long(\n            final_oof\n        )\n    )\n\n    pd.DataFrame(\n        [\n            {\n                "model":\n                    "FINAL_EXP05_5SEED_AVERAGE",\n\n                "max_duplicate_target_discrepancy":\n                    discrepancy,\n            }\n        ]\n    ).to_csv(\n        METRICS_DIR\n        / "FINAL_EXP05_target_consistency.csv",\n        index=False,\n    )\n\n    return (\n        final_oof,\n        final_metrics,\n        seed_metrics,\n        seed_summary,\n    )\n\n\ndef average_oof_predictions(\n    seed_oofs,\n):\n\n    keys = [\n        "fipsCode",\n        "origin_timestamp",\n        "horizon_h",\n        "target_timestamp",\n        "fold",\n    ]\n\n    merged = None\n    pred_cols = []\n\n    for seed in FINAL_SEEDS:\n\n        current = (\n            seed_oofs[\n                seed\n            ][\n                keys\n                + [\n                    "actual_osi",\n                    "predicted_osi",\n                ]\n            ]\n            .rename(\n                columns={\n                    "actual_osi":\n                        f"actual_seed_{seed}",\n\n                    "predicted_osi":\n                        f"pred_seed_{seed}",\n                }\n            )\n        )\n\n        pred_cols.append(\n            f"pred_seed_{seed}"\n        )\n\n        if merged is None:\n\n            merged = current\n\n        else:\n\n            merged = merged.merge(\n                current,\n                on=keys,\n                how="inner",\n                validate="one_to_one",\n            )\n\n    actual_cols = [\n        f"actual_seed_{seed}"\n        for seed in FINAL_SEEDS\n    ]\n\n    canonical_actual = (\n        merged[\n            actual_cols[\n                0\n            ]\n        ]\n        .to_numpy(float)\n    )\n\n    for col in (\n        actual_cols[\n            1:\n        ]\n    ):\n\n        difference = np.max(\n            np.abs(\n                canonical_actual\n                - merged[\n                    col\n                ]\n                .to_numpy(float)\n            )\n        )\n\n        if difference > 1e-5:\n\n            raise RuntimeError(\n                "Actual OOF targets differ materially "\n                f"across seeds: max diff={difference}"\n            )\n\n    matrix = (\n        merged[\n            pred_cols\n        ]\n        .to_numpy(float)\n    )\n\n    result = (\n        merged[\n            keys\n        ]\n        .copy()\n    )\n\n    result[\n        "actual_osi"\n    ] = canonical_actual\n\n    result[\n        "predicted_osi"\n    ] = np.mean(\n        matrix,\n        axis=1,\n    )\n\n    result[\n        "seed_prediction_std"\n    ] = np.std(\n        matrix,\n        axis=1,\n        ddof=1,\n    )\n\n    return result\n\n\n# ================================================================\n# 9. FINAL FULL-DATA TRAINING / SUBMISSION\n# ================================================================\n\ndef build_submission_from_prediction(\n    test_arrays,\n    prediction,\n):\n\n    template = pd.read_csv(\n        TEMPLATE_PATH\n    )\n\n    output = template.copy()\n\n    official_nan = (\n        template[\n            TARGET_COLS\n        ]\n        .isna()\n        .to_numpy()\n    )\n\n    prediction_map = {}\n\n    for county_i, fips in enumerate(\n        test_arrays[\n            "county_ids"\n        ]\n    ):\n\n        for origin_i in range(\n            N_FUTURE\n        ):\n\n            key = (\n                int(\n                    fips\n                ),\n                pd.Timestamp(\n                    test_arrays[\n                        "timestamps"\n                    ][\n                        county_i,\n                        origin_i,\n                    ]\n                ),\n            )\n\n            prediction_map[\n                key\n            ] = prediction[\n                county_i,\n                origin_i,\n                :\n            ]\n\n    origin_times = pd.to_datetime(\n        template[\n            "timestamp_et"\n        ]\n    )\n\n    values = np.full(\n        (\n            len(\n                template\n            ),\n            len(\n                HORIZONS\n            ),\n        ),\n        np.nan,\n        dtype=float,\n    )\n\n    for i in range(\n        len(\n            template\n        )\n    ):\n\n        key = (\n            int(\n                template.at[\n                    i,\n                    "fipsCode",\n                ]\n            ),\n            pd.Timestamp(\n                origin_times.iloc[\n                    i\n                ]\n            ),\n        )\n\n        vector = prediction_map[\n            key\n        ]\n\n        for j in range(\n            len(\n                HORIZONS\n            )\n        ):\n\n            if not official_nan[\n                i,\n                j,\n            ]:\n\n                values[\n                    i,\n                    j,\n                ] = float(\n                    vector[\n                        j\n                    ]\n                )\n\n    output[\n        TARGET_COLS\n    ] = values\n\n    return output\n\n\ndef validate_submission(\n    output,\n    label,\n):\n\n    template = pd.read_csv(\n        TEMPLATE_PATH\n    )\n\n    assert output.shape == template.shape\n    assert list(output.columns) == list(template.columns)\n\n    identifiers = [\n        "fipsCode",\n        "countyName",\n        "stateAbbr",\n        "timestamp_et",\n    ]\n\n    assert (\n        output[\n            identifiers\n        ]\n        .equals(\n            template[\n                identifiers\n            ]\n        )\n    )\n\n    assert np.array_equal(\n        output[\n            TARGET_COLS\n        ]\n        .isna()\n        .to_numpy(),\n        template[\n            TARGET_COLS\n        ]\n        .isna()\n        .to_numpy(),\n    )\n\n    for col, expected in (\n        EXPECTED_TEMPLATE_NANS.items()\n    ):\n\n        assert (\n            output[\n                col\n            ]\n            .isna()\n            .sum()\n            == expected\n        )\n\n    finite = (\n        output[\n            TARGET_COLS\n        ]\n        .to_numpy(float)\n    )\n\n    finite = finite[\n        np.isfinite(\n            finite\n        )\n    ]\n\n    assert np.all(\n        finite >= 0\n    )\n\n    discrepancy = (\n        duplicate_target_discrepancy_submission(\n            output\n        )\n    )\n\n    if discrepancy > 1e-6:\n\n        raise RuntimeError(\n            f"{label} lost physical-target consistency: "\n            f"{discrepancy}"\n        )\n\n    return discrepancy\n\n\ndef run_final_submission(\n    train,\n    test,\n    force=False,\n):\n\n    final_path = (\n        SUBMISSION_DIR\n        / "FINAL_PRIMARY_EXP05_5SEED_AVERAGE.csv"\n    )\n\n    if (\n        final_path.exists()\n        and not force\n    ):\n\n        print(\n            "\\nReusing cached final primary submission."\n        )\n\n        output = pd.read_csv(\n            final_path\n        )\n\n        discrepancy = (\n            validate_submission(\n                output,\n                "cached final submission",\n            )\n        )\n\n        return (\n            output,\n            discrepancy,\n        )\n\n    print(\n        "\\n"\n        + "=" * 78\n    )\n\n    print(\n        "FINAL FULL-DATA TRAINING: EXP05 FIVE-SEED AVERAGE"\n    )\n\n    print(\n        "=" * 78\n    )\n\n    all_train_ids = np.sort(\n        train[\n            "fipsCode"\n        ].unique()\n    )\n\n    all_test_ids = np.sort(\n        test[\n            "fipsCode"\n        ].unique()\n    )\n\n    (\n        history_scaler,\n        future_scaler,\n    ) = fit_final_scalers(\n        train,\n        all_train_ids,\n    )\n\n    train_arrays = (\n        build_final_county_arrays(\n            train,\n            all_train_ids,\n            history_scaler,\n            future_scaler,\n            with_targets=True,\n        )\n    )\n\n    test_arrays = (\n        build_final_county_arrays(\n            test,\n            all_test_ids,\n            history_scaler,\n            future_scaler,\n            with_targets=False,\n        )\n    )\n\n    # Save scalers once; they are data-dependent, not seed-dependent.\n    import pickle\n\n    with open(\n        MODEL_DIR\n        / "FINAL_EXP05_scalers.pkl",\n        "wb",\n    ) as f:\n\n        pickle.dump(\n            {\n                "history_scaler":\n                    history_scaler,\n\n                "future_scaler":\n                    future_scaler,\n\n                "history_features":\n                    HISTORY_COLS,\n\n                "future_features":\n                    FUTURE_COLS,\n            },\n            f,\n        )\n\n    seed_submissions = []\n\n    for seed in FINAL_SEEDS:\n\n        seed_path = (\n            SUBMISSION_DIR\n            / f"FINAL_EXP05_seed{seed}.csv"\n        )\n\n        model_path = (\n            MODEL_DIR\n            / f"FINAL_EXP05_seed{seed}_model.pt"\n        )\n\n        history_path = (\n            METRICS_DIR\n            / f"FINAL_EXP05_seed{seed}_full_history.csv"\n        )\n\n        if (\n            seed_path.exists()\n            and model_path.exists()\n            and history_path.exists()\n            and not force\n        ):\n\n            print(\n                f"Reusing full-data final seed {seed}."\n            )\n\n            seed_output = pd.read_csv(\n                seed_path\n            )\n\n            validate_submission(\n                seed_output,\n                f"seed {seed}",\n            )\n\n            seed_submissions.append(\n                seed_output\n            )\n\n            continue\n\n        print(\n            f"\\nTraining final seed {seed}..."\n        )\n\n        model, history = (\n            final_train_model(\n                train_arrays,\n                seed\n                + 10000,\n            )\n        )\n\n        prediction = (\n            final_predict(\n                model,\n                test_arrays,\n            )\n        )\n\n        seed_output = (\n            build_submission_from_prediction(\n                test_arrays,\n                prediction,\n            )\n        )\n\n        validate_submission(\n            seed_output,\n            f"seed {seed}",\n        )\n\n        seed_output.to_csv(\n            seed_path,\n            index=False,\n        )\n\n        history.to_csv(\n            history_path,\n            index=False,\n        )\n\n        torch.save(\n            {\n                "state_dict":\n                    model.state_dict(),\n\n                "seed":\n                    seed\n                    + 10000,\n\n                "base_seed":\n                    seed,\n\n                "architecture":\n                    "EXP05_CNNV2_FIXED_HISTORY",\n\n                "history_features":\n                    HISTORY_COLS,\n\n                "future_features":\n                    FUTURE_COLS,\n\n                "horizons":\n                    HORIZONS,\n            },\n            model_path,\n        )\n\n        seed_submissions.append(\n            seed_output\n        )\n\n        del model\n\n        if torch.cuda.is_available():\n            torch.cuda.empty_cache()\n\n    template = pd.read_csv(\n        TEMPLATE_PATH\n    )\n\n    final_output = (\n        template.copy()\n    )\n\n    official_nan = (\n        template[\n            TARGET_COLS\n        ]\n        .isna()\n        .to_numpy()\n    )\n\n    for col in TARGET_COLS:\n\n        matrix = np.column_stack(\n            [\n                submission[\n                    col\n                ]\n                .to_numpy(float)\n\n                for submission\n                in seed_submissions\n            ]\n        )\n\n        averaged = np.nanmean(\n            matrix,\n            axis=1,\n        )\n\n        all_nan = np.all(\n            ~np.isfinite(\n                matrix\n            ),\n            axis=1,\n        )\n\n        averaged[\n            all_nan\n        ] = np.nan\n\n        final_output[\n            col\n        ] = averaged\n\n    final_output[\n        TARGET_COLS\n    ] = (\n        final_output[\n            TARGET_COLS\n        ]\n        .mask(\n            official_nan\n        )\n    )\n\n    discrepancy = (\n        validate_submission(\n            final_output,\n            "final five-seed average",\n        )\n    )\n\n    final_output.to_csv(\n        final_path,\n        index=False,\n    )\n\n    pd.DataFrame(\n        [\n            {\n                "submission":\n                    final_path.name,\n\n                "n_seeds":\n                    len(\n                        FINAL_SEEDS\n                    ),\n\n                "seeds":\n                    ",".join(\n                        str(seed)\n                        for seed\n                        in FINAL_SEEDS\n                    ),\n\n                "max_duplicate_target_discrepancy":\n                    discrepancy,\n\n                **{\n                    f"nan_{col}":\n                        int(\n                            final_output[\n                                col\n                            ]\n                            .isna()\n                            .sum()\n                        )\n                    for col\n                    in TARGET_COLS\n                },\n            }\n        ]\n    ).to_csv(\n        METRICS_DIR\n        / "FINAL_submission_validation.csv",\n        index=False,\n    )\n\n    return (\n        final_output,\n        discrepancy,\n    )\n\n\n# ================================================================\n# 10. FINAL VS BASELINE COMPARISON / BOOTSTRAP\n# ================================================================\n\ndef compare_baseline_final(\n    baseline_oof,\n    final_oof,\n):\n\n    keys = [\n        "fipsCode",\n        "origin_timestamp",\n        "horizon_h",\n        "target_timestamp",\n        "fold",\n    ]\n\n    merged = (\n        baseline_oof[\n            keys\n            + [\n                "actual_osi",\n                "predicted_osi",\n            ]\n        ]\n        .rename(\n            columns={\n                "actual_osi":\n                    "actual_baseline",\n\n                "predicted_osi":\n                    "pred_baseline",\n            }\n        )\n        .merge(\n            final_oof[\n                keys\n                + [\n                    "actual_osi",\n                    "predicted_osi",\n                    "seed_prediction_std",\n                ]\n            ]\n            .rename(\n                columns={\n                    "actual_osi":\n                        "actual_final",\n\n                    "predicted_osi":\n                        "pred_final",\n                }\n            ),\n            on=keys,\n            how="inner",\n            validate="one_to_one",\n        )\n    )\n\n    target_diff = float(\n        np.max(\n            np.abs(\n                merged[\n                    "actual_baseline"\n                ]\n                - merged[\n                    "actual_final"\n                ]\n            )\n        )\n    )\n\n    if target_diff > 1e-5:\n\n        raise RuntimeError(\n            "Baseline/final actual targets disagree "\n            f"materially: {target_diff}"\n        )\n\n    merged[\n        "actual_osi"\n    ] = merged[\n        "actual_final"\n    ]\n\n    rows = []\n\n    for horizon in HORIZONS:\n\n        subset = merged[\n            merged[\n                "horizon_h"\n            ]\n            == horizon\n        ]\n\n        baseline_rmse = (\n            calculate_metrics(\n                subset[\n                    "actual_osi"\n                ],\n                subset[\n                    "pred_baseline"\n                ],\n            )[\n                "RMSE"\n            ]\n        )\n\n        final_rmse = (\n            calculate_metrics(\n                subset[\n                    "actual_osi"\n                ],\n                subset[\n                    "pred_final"\n                ],\n            )[\n                "RMSE"\n            ]\n        )\n\n        rows.append(\n            {\n                "horizon":\n                    f"t+{horizon}h",\n\n                "baseline_RMSE":\n                    baseline_rmse,\n\n                "final_RMSE":\n                    final_rmse,\n\n                "RMSE_reduction_pct":\n                    100.0\n                    * (\n                        baseline_rmse\n                        - final_rmse\n                    )\n                    / baseline_rmse,\n            }\n        )\n\n    comparison = pd.DataFrame(\n        rows\n    )\n\n    comparison.to_csv(\n        METRICS_DIR\n        / "FINAL_vs_INITIAL_CNNLSTM.csv",\n        index=False,\n    )\n\n    bootstrap = (\n        bootstrap_final_vs_baseline(\n            merged\n        )\n    )\n\n    bootstrap.to_csv(\n        METRICS_DIR\n        / "FINAL_vs_INITIAL_CNNLSTM_county_bootstrap.csv",\n        index=False,\n    )\n\n    save_final_diagnostics(\n        merged,\n        comparison,\n    )\n\n    return (\n        merged,\n        comparison,\n        bootstrap,\n    )\n\n\ndef bootstrap_final_vs_baseline(\n    merged,\n):\n\n    rng = np.random.default_rng(\n        BOOTSTRAP_SEED\n    )\n\n    rows = []\n\n    for horizon in HORIZONS:\n\n        h = merged[\n            merged[\n                "horizon_h"\n            ]\n            == horizon\n        ]\n\n        county_rows = []\n\n        for fips, county in (\n            h.groupby(\n                "fipsCode"\n            )\n        ):\n\n            actual = (\n                county[\n                    "actual_osi"\n                ]\n                .to_numpy(float)\n            )\n\n            county_rows.append(\n                {\n                    "fipsCode":\n                        int(\n                            fips\n                        ),\n\n                    "n":\n                        len(\n                            county\n                        ),\n\n                    "sse_baseline":\n                        float(\n                            np.sum(\n                                (\n                                    county[\n                                        "pred_baseline"\n                                    ]\n                                    .to_numpy(float)\n                                    - actual\n                                ) ** 2\n                            )\n                        ),\n\n                    "sse_final":\n                        float(\n                            np.sum(\n                                (\n                                    county[\n                                        "pred_final"\n                                    ]\n                                    .to_numpy(float)\n                                    - actual\n                                ) ** 2\n                            )\n                        ),\n                }\n            )\n\n        county = pd.DataFrame(\n            county_rows\n        )\n\n        n_counties = len(\n            county\n        )\n\n        counts = (\n            county[\n                "n"\n            ]\n            .to_numpy(float)\n        )\n\n        sse_base = (\n            county[\n                "sse_baseline"\n            ]\n            .to_numpy(float)\n        )\n\n        sse_final = (\n            county[\n                "sse_final"\n            ]\n            .to_numpy(float)\n        )\n\n        improvements = np.empty(\n            BOOTSTRAP_REPS,\n            dtype=float,\n        )\n\n        for b in range(\n            BOOTSTRAP_REPS\n        ):\n\n            idx = rng.integers(\n                0,\n                n_counties,\n                size=n_counties,\n            )\n\n            total_n = np.sum(\n                counts[\n                    idx\n                ]\n            )\n\n            rmse_base = np.sqrt(\n                np.sum(\n                    sse_base[\n                        idx\n                    ]\n                )\n                / total_n\n            )\n\n            rmse_final = np.sqrt(\n                np.sum(\n                    sse_final[\n                        idx\n                    ]\n                )\n                / total_n\n            )\n\n            improvements[\n                b\n            ] = (\n                100.0\n                * (\n                    rmse_base\n                    - rmse_final\n                )\n                / rmse_base\n            )\n\n        point_base = np.sqrt(\n            np.sum(\n                sse_base\n            )\n            / np.sum(\n                counts\n            )\n        )\n\n        point_final = np.sqrt(\n            np.sum(\n                sse_final\n            )\n            / np.sum(\n                counts\n            )\n        )\n\n        rows.append(\n            {\n                "horizon":\n                    f"t+{horizon}h",\n\n                "point_RMSE_reduction_pct":\n                    100.0\n                    * (\n                        point_base\n                        - point_final\n                    )\n                    / point_base,\n\n                "bootstrap_mean_pct":\n                    float(\n                        np.mean(\n                            improvements\n                        )\n                    ),\n\n                "CI_2p5_pct":\n                    float(\n                        np.percentile(\n                            improvements,\n                            2.5,\n                        )\n                    ),\n\n                "CI_97p5_pct":\n                    float(\n                        np.percentile(\n                            improvements,\n                            97.5,\n                        )\n                    ),\n\n                "probability_improvement_gt_0":\n                    float(\n                        np.mean(\n                            improvements\n                            > 0\n                        )\n                    ),\n            }\n        )\n\n    return pd.DataFrame(\n        rows\n    )\n\n\n# ================================================================\n# 11. VISUALIZATIONS\n# ================================================================\n\ndef save_final_diagnostics(\n    merged,\n    comparison,\n):\n\n    # ------------------------------------------------------------\n    # RMSE comparison\n    # ------------------------------------------------------------\n\n    fig, ax = plt.subplots(\n        figsize=(10, 6)\n    )\n\n    x = np.arange(\n        len(\n            HORIZONS\n        )\n    )\n\n    width = 0.34\n\n    ax.bar(\n        x\n        - width\n        / 2,\n        comparison[\n            "baseline_RMSE"\n        ],\n        width=width,\n        label="Initial cleaned CNN-LSTM (EXP03)",\n    )\n\n    ax.bar(\n        x\n        + width\n        / 2,\n        comparison[\n            "final_RMSE"\n        ],\n        width=width,\n        label="Final EXP05 five-seed average",\n    )\n\n    ax.set_xticks(\n        x\n    )\n\n    ax.set_xticklabels(\n        [\n            "t+1h",\n            "t+6h",\n            "t+24h",\n            "t+48h",\n        ]\n    )\n\n    ax.set_ylabel(\n        "Pooled OOF RMSE"\n    )\n\n    ax.set_title(\n        "Initial CNN-LSTM vs final five-seed CNN-v2"\n    )\n\n    ax.grid(\n        axis="y",\n        alpha=0.25,\n    )\n\n    ax.legend()\n\n    fig.tight_layout()\n\n    fig.savefig(\n        FIGURE_DIR\n        / "FINAL_vs_INITIAL_RMSE.png",\n        dpi=220,\n        bbox_inches="tight",\n    )\n\n    plt.close(\n        fig\n    )\n\n    # ------------------------------------------------------------\n    # Actual vs predicted scatter per horizon\n    # ------------------------------------------------------------\n\n    for horizon in HORIZONS:\n\n        subset = merged[\n            merged[\n                "horizon_h"\n            ]\n            == horizon\n        ]\n\n        fig, ax = plt.subplots(\n            figsize=(7, 7)\n        )\n\n        ax.scatter(\n            subset[\n                "actual_osi"\n            ],\n            subset[\n                "pred_final"\n            ],\n            s=8,\n            alpha=0.20,\n        )\n\n        limit = float(\n            max(\n                subset[\n                    "actual_osi"\n                ].max(),\n                subset[\n                    "pred_final"\n                ].max(),\n            )\n        )\n\n        ax.plot(\n            [\n                0,\n                limit,\n            ],\n            [\n                0,\n                limit,\n            ],\n            linestyle="--",\n            linewidth=1,\n        )\n\n        ax.set_xlabel(\n            "Actual OSI"\n        )\n\n        ax.set_ylabel(\n            "Final predicted OSI"\n        )\n\n        ax.set_title(\n            f"Final five-seed CNN-v2: t+{horizon}h"\n        )\n\n        ax.grid(\n            alpha=0.20\n        )\n\n        fig.tight_layout()\n\n        fig.savefig(\n            FIGURE_DIR\n            / f"FINAL_scatter_{horizon}h.png",\n            dpi=220,\n            bbox_inches="tight",\n        )\n\n        plt.close(\n            fig\n        )\n\n    # ------------------------------------------------------------\n    # County error comparison\n    # ------------------------------------------------------------\n\n    county = (\n        merged\n        .assign(\n            sq_baseline=(\n                merged[\n                    "pred_baseline"\n                ]\n                - merged[\n                    "actual_osi"\n                ]\n            ) ** 2,\n\n            sq_final=(\n                merged[\n                    "pred_final"\n                ]\n                - merged[\n                    "actual_osi"\n                ]\n            ) ** 2,\n        )\n        .groupby(\n            "fipsCode",\n            as_index=False,\n        )\n        .agg(\n            mse_baseline=(\n                "sq_baseline",\n                "mean",\n            ),\n            mse_final=(\n                "sq_final",\n                "mean",\n            ),\n            peak_actual=(\n                "actual_osi",\n                "max",\n            ),\n        )\n    )\n\n    county[\n        "RMSE_baseline"\n    ] = np.sqrt(\n        county[\n            "mse_baseline"\n        ]\n    )\n\n    county[\n        "RMSE_final"\n    ] = np.sqrt(\n        county[\n            "mse_final"\n        ]\n    )\n\n    county = county.sort_values(\n        "RMSE_final",\n        ascending=False,\n    )\n\n    county.to_csv(\n        METRICS_DIR\n        / "FINAL_county_error_summary.csv",\n        index=False,\n    )\n\n    top = (\n        county\n        .head(\n            15\n        )\n        .iloc[\n            ::-1\n        ]\n    )\n\n    y = np.arange(\n        len(\n            top\n        )\n    )\n\n    height = 0.36\n\n    fig, ax = plt.subplots(\n        figsize=(11, 8)\n    )\n\n    ax.barh(\n        y\n        - height\n        / 2,\n        top[\n            "RMSE_baseline"\n        ],\n        height=height,\n        label="Initial CNN-LSTM",\n    )\n\n    ax.barh(\n        y\n        + height\n        / 2,\n        top[\n            "RMSE_final"\n        ],\n        height=height,\n        label="Final five-seed CNN-v2",\n    )\n\n    ax.set_yticks(\n        y\n    )\n\n    ax.set_yticklabels(\n        top[\n            "fipsCode"\n        ]\n        .astype(str)\n    )\n\n    ax.set_xlabel(\n        "County pooled RMSE"\n    )\n\n    ax.set_title(\n        "Highest-error final-model counties"\n    )\n\n    ax.legend()\n\n    ax.grid(\n        axis="x",\n        alpha=0.25,\n    )\n\n    fig.tight_layout()\n\n    fig.savefig(\n        FIGURE_DIR\n        / "FINAL_worst_counties.png",\n        dpi=220,\n        bbox_inches="tight",\n    )\n\n    plt.close(\n        fig\n    )\n\n    # ------------------------------------------------------------\n    # Trajectories for six worst final counties (t+1 target timeline)\n    # ------------------------------------------------------------\n\n    for fips in (\n        county\n        .head(\n            6\n        )[\n            "fipsCode"\n        ]\n        .astype(int)\n    ):\n\n        subset = (\n            merged[\n                (\n                    merged[\n                        "fipsCode"\n                    ]\n                    == fips\n                )\n                & (\n                    merged[\n                        "horizon_h"\n                    ]\n                    == 1\n                )\n            ]\n            .sort_values(\n                "target_timestamp"\n            )\n        )\n\n        fig, ax = plt.subplots(\n            figsize=(12, 5)\n        )\n\n        ax.plot(\n            subset[\n                "target_timestamp"\n            ],\n            subset[\n                "actual_osi"\n            ],\n            linewidth=2,\n            label="Actual",\n        )\n\n        ax.plot(\n            subset[\n                "target_timestamp"\n            ],\n            subset[\n                "pred_baseline"\n            ],\n            label="Initial CNN-LSTM",\n        )\n\n        ax.plot(\n            subset[\n                "target_timestamp"\n            ],\n            subset[\n                "pred_final"\n            ],\n            label="Final five-seed CNN-v2",\n        )\n\n        ax.set_ylabel(\n            "OSI"\n        )\n\n        ax.set_title(\n            f"County FIPS {fips}"\n        )\n\n        ax.legend()\n\n        ax.grid(\n            alpha=0.25\n        )\n\n        fig.autofmt_xdate()\n        fig.tight_layout()\n\n        fig.savefig(\n            FIGURE_DIR\n            / f"FINAL_trajectory_{fips}.png",\n            dpi=220,\n            bbox_inches="tight",\n        )\n\n        plt.close(\n            fig\n        )\n\n    # ------------------------------------------------------------\n    # Severe-event diagnostics\n    # ------------------------------------------------------------\n\n    thresholds = [\n        0.01,\n        0.05,\n        0.10,\n        0.20,\n    ]\n\n    rows = []\n\n    for threshold in thresholds:\n\n        subset = merged[\n            merged[\n                "actual_osi"\n            ]\n            >= threshold\n        ]\n\n        actual = (\n            subset[\n                "actual_osi"\n            ]\n            .to_numpy(float)\n        )\n\n        pred = (\n            subset[\n                "pred_final"\n            ]\n            .to_numpy(float)\n        )\n\n        rows.append(\n            {\n                "threshold":\n                    threshold,\n\n                "n":\n                    len(\n                        subset\n                    ),\n\n                "underprediction_rate":\n                    float(\n                        np.mean(\n                            pred\n                            < actual\n                        )\n                    ),\n\n                "predicted_to_actual_sum_ratio":\n                    float(\n                        np.sum(\n                            pred\n                        )\n                        / np.sum(\n                            actual\n                        )\n                    ),\n            }\n        )\n\n    severe = pd.DataFrame(\n        rows\n    )\n\n    severe.to_csv(\n        METRICS_DIR\n        / "FINAL_severe_event_diagnostics.csv",\n        index=False,\n    )\n\n    fig, ax = plt.subplots(\n        figsize=(8, 5)\n    )\n\n    ax.plot(\n        severe[\n            "threshold"\n        ],\n        severe[\n            "predicted_to_actual_sum_ratio"\n        ],\n        marker="o",\n    )\n\n    ax.axhline(\n        1.0,\n        linestyle="--",\n        linewidth=1,\n    )\n\n    ax.set_xlabel(\n        "Actual OSI threshold"\n    )\n\n    ax.set_ylabel(\n        "Predicted / actual severity ratio"\n    )\n\n    ax.set_title(\n        "Final model severe-event magnitude calibration"\n    )\n\n    ax.grid(\n        alpha=0.25\n    )\n\n    fig.tight_layout()\n\n    fig.savefig(\n        FIGURE_DIR\n        / "FINAL_severe_event_calibration.png",\n        dpi=220,\n        bbox_inches="tight",\n    )\n\n    plt.close(\n        fig\n    )\n\n\ndef plot_final_seed_robustness(\n    seed_metrics,\n    seed_summary,\n):\n\n    fig, ax = plt.subplots(\n        figsize=(10, 6)\n    )\n\n    horizon_order = [\n        "t+1h",\n        "t+6h",\n        "t+24h",\n        "t+48h",\n    ]\n\n    for seed in FINAL_SEEDS:\n\n        subset = (\n            seed_metrics[\n                seed_metrics[\n                    "seed"\n                ]\n                == seed\n            ]\n            .set_index(\n                "horizon"\n            )\n            .reindex(\n                horizon_order\n            )\n        )\n\n        ax.plot(\n            horizon_order,\n            subset[\n                "RMSE"\n            ],\n            marker="o",\n            label=f"seed {seed}",\n        )\n\n    ax.set_ylabel(\n        "Pooled OOF RMSE"\n    )\n\n    ax.set_title(\n        "Final EXP05 seed-to-seed variability"\n    )\n\n    ax.grid(\n        alpha=0.25\n    )\n\n    ax.legend()\n\n    fig.tight_layout()\n\n    fig.savefig(\n        FIGURE_DIR\n        / "FINAL_per_seed_RMSE.png",\n        dpi=220,\n        bbox_inches="tight",\n    )\n\n    plt.close(\n        fig\n    )\n\n    fig, ax = plt.subplots(\n        figsize=(9, 6)\n    )\n\n    subset = (\n        seed_summary\n        .set_index(\n            "horizon"\n        )\n        .reindex(\n            horizon_order\n        )\n    )\n\n    x = np.arange(\n        len(\n            horizon_order\n        )\n    )\n\n    ax.errorbar(\n        x,\n        subset[\n            "mean_RMSE"\n        ],\n        yerr=subset[\n            "std_RMSE"\n        ],\n        marker="o",\n        capsize=5,\n    )\n\n    ax.set_xticks(\n        x\n    )\n\n    ax.set_xticklabels(\n        horizon_order\n    )\n\n    ax.set_ylabel(\n        "RMSE mean ± seed SD"\n    )\n\n    ax.set_title(\n        "Final EXP05 seed robustness"\n    )\n\n    ax.grid(\n        axis="y",\n        alpha=0.25,\n    )\n\n    fig.tight_layout()\n\n    fig.savefig(\n        FIGURE_DIR\n        / "FINAL_seed_mean_std_RMSE.png",\n        dpi=220,\n        bbox_inches="tight",\n    )\n\n    plt.close(\n        fig\n    )\n\n\n# ================================================================\n# 12. MAIN\n# ================================================================\n\ndef run_pipeline(args):\n\n    configure_determinism()\n    set_seed(\n        FOLD_SEED\n    )\n\n    require_inputs()\n\n    write_environment_snapshot()\n    write_run_config(\n        args\n    )\n\n    train, test = load_data()\n\n    audit_inputs(\n        train,\n        test,\n    )\n\n    folds = create_or_verify_folds(\n        train\n    )\n\n    write_input_manifest()\n\n    if args.mode == "audit":\n\n        print(\n            "\\nAudit-only mode complete."\n        )\n\n        return\n\n    baseline_oof = None\n    final_oof = None\n\n    if args.mode in [\n        "compare",\n        "all",\n    ]:\n\n        (\n            baseline_oof,\n            baseline_metrics,\n        ) = run_baseline_cv(\n            train,\n            folds,\n            force=args.force,\n        )\n\n        (\n            final_oof,\n            final_metrics,\n            seed_metrics,\n            seed_summary,\n        ) = run_final_cv(\n            train,\n            folds,\n            force=args.force,\n        )\n\n        plot_final_seed_robustness(\n            seed_metrics,\n            seed_summary,\n        )\n\n        (\n            merged,\n            comparison,\n            bootstrap,\n        ) = compare_baseline_final(\n            baseline_oof,\n            final_oof,\n        )\n\n        print(\n            "\\n"\n            + "=" * 78\n        )\n\n        print(\n            "FINAL MODEL VS INITIAL CLEANED CNN-LSTM"\n        )\n\n        print(\n            "=" * 78\n        )\n\n        print(\n            comparison\n            .round(6)\n            .to_string(\n                index=False\n            )\n        )\n\n        print(\n            "\\nCounty-bootstrap RMSE reduction:"\n        )\n\n        print(\n            bootstrap\n            .round(4)\n            .to_string(\n                index=False\n            )\n        )\n\n    if args.mode in [\n        "submit",\n        "all",\n    ]:\n\n        (\n            final_submission,\n            discrepancy,\n        ) = run_final_submission(\n            train,\n            test,\n            force=args.force,\n        )\n\n        print(\n            "\\n"\n            + "=" * 78\n        )\n\n        print(\n            "FINAL PRIMARY SUBMISSION READY"\n        )\n\n        print(\n            "=" * 78\n        )\n\n        print(\n            SUBMISSION_DIR\n            / "FINAL_PRIMARY_EXP05_5SEED_AVERAGE.csv"\n        )\n\n        print(\n            "Maximum duplicate-target discrepancy:",\n            f"{discrepancy:.10f}"\n        )\n\n    print(\n        "\\nArtifacts saved under:"\n    )\n\n    print(\n        FINAL_ROOT\n    )\n\n\ndef parse_args():\n\n    parser = argparse.ArgumentParser(\n        description=(\n            "Final reproducible INFORMS 2026 "\n            "Data Challenge pipeline."\n        )\n    )\n\n    parser.add_argument(\n        "--mode",\n        choices=[\n            "audit",\n            "compare",\n            "submit",\n            "all",\n        ],\n        default="all",\n        help=(\n            "audit: data checks only; "\n            "compare: baseline + final CV; "\n            "submit: full final training/submission; "\n            "all: everything."\n        ),\n    )\n\n    parser.add_argument(\n        "--force",\n        action="store_true",\n        help=(\n            "Intentionally rerun and overwrite "\n            "cached final artifacts."\n        ),\n    )\n\n    # parse_known_args keeps the script friendly to Spyder/IPython,\n    # which may inject additional command-line arguments.\n    args, _ = parser.parse_known_args()\n    return args\n\n\ndef main():\n\n    args = parse_args()\n\n    timestamp = (\n        datetime.now()\n        .strftime(\n            "%Y%m%d_%H%M%S"\n        )\n    )\n\n    log_path = (\n        LOG_DIR\n        / f"final_pipeline_{timestamp}.txt"\n    )\n\n    original_stdout = sys.stdout\n    original_stderr = sys.stderr\n\n    with open(\n        log_path,\n        "w",\n        encoding="utf-8",\n    ) as log_file:\n\n        sys.stdout = Tee(\n            original_stdout,\n            log_file,\n        )\n\n        sys.stderr = Tee(\n            original_stderr,\n            log_file,\n        )\n\n        try:\n\n            print(\n                "=" * 78\n            )\n\n            print(\n                "INFORMS 2026 FINAL REPRODUCIBLE PIPELINE"\n            )\n\n            print(\n                "=" * 78\n            )\n\n            print(\n                "Mode:",\n                args.mode\n            )\n\n            print(\n                "Device:",\n                DEVICE\n            )\n\n            if torch.cuda.is_available():\n\n                print(\n                    "GPU:",\n                    torch.cuda.get_device_name(\n                        0\n                    )\n                )\n\n            print(\n                "Primary final model:",\n                "EXP05 five-seed average",\n            )\n\n            run_pipeline(\n                args\n            )\n\n        except Exception:\n\n            print(\n                "\\nFINAL PIPELINE FAILED:\\n"\n            )\n\n            traceback.print_exc()\n\n            raise\n\n        finally:\n\n            print(\n                "\\nConsole log:"\n            )\n\n            print(\n                log_path\n            )\n\n            sys.stdout = (\n                original_stdout\n            )\n\n            sys.stderr = (\n                original_stderr\n            )\n\n\nif __name__ == "__main__":\n    raise SystemExit(\n        "This file is a support module. Run phase14_final_trial22_submission.py instead."\n    )\n'

base = types.ModuleType("embedded_final_reproducible_pipeline")
base.__file__ = __file__
base.__dict__["__file__"] = __file__
base.__dict__["__name__"] = "embedded_final_reproducible_pipeline"

try:
    exec(
        compile(
            EMBEDDED_BASE_SOURCE,
            "<embedded_final_reproducible_pipeline>",
            "exec",
        ),
        base.__dict__,
    )
except Exception as exc:
    raise RuntimeError(
        "Failed to initialize the embedded final reproducible pipeline."
    ) from exc


# =====================================================================
# CONFIG
# =====================================================================
ROOT = Path(__file__).resolve().parent

OUT = (
    ROOT
    / "final_trial22_state_uot_submission"
)
OUT.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_DIR = OUT / "models"
SUBMISSION_DIR = OUT / "submissions"
METRICS_DIR = OUT / "metrics"
FIG_DIR = OUT / "figures"
PROVENANCE_DIR = OUT / "provenance"
RAW_PRED_DIR = OUT / "raw_predictions"
COMPETITION_DIR = OUT / "competition_ready"

for p in [
    MODEL_DIR,
    SUBMISSION_DIR,
    METRICS_DIR,
    FIG_DIR,
    PROVENANCE_DIR,
    RAW_PRED_DIR,
    COMPETITION_DIR,
]:
    p.mkdir(
        parents=True,
        exist_ok=True,
    )

BASE_SEEDS = [
    0,
    17,
    42,
    73,
    123,
]

FULL_TRAIN_SEED_OFFSET = 10000

EPOCHS = 120
BATCH_SIZE = 16
LR = 0.0009448516088066339

WARMUP_EPOCHS = 20
RAMP_EPOCHS = 30

UOT_REACH = 1.9492145906383787
UOT_BLUR = 0.11988343825386838
UOT_DIAMETER = math.sqrt(2.0)
FROZEN_LAMBDA_UOT = 0.0012262539110903918
MASS_EPS = 1e-8

STATE_NAMES = {
    18: "IN",
    39: "OH",
    42: "PA",
    54: "WV",
}

RESUME_COMPLETED_SEEDS = True

FINAL_SUBMISSION_NAME = (
    "FINAL_TRIAL22_STATE_UOT_5SEED_AVERAGE.csv"
)



# =====================================================================
# FROZEN TRIAL-22 CONFIG / SAFE RESUME
# =====================================================================

def frozen_config_dict():
    return {
        "model": "Trial22_StateUOT_FinalCNNV2",
        "epochs": int(EPOCHS),
        "batch_size": int(BATCH_SIZE),
        "learning_rate": float(LR),
        "warmup_epochs": int(WARMUP_EPOCHS),
        "ramp_epochs": int(RAMP_EPOCHS),
        "reach": float(UOT_REACH),
        "blur": float(UOT_BLUR),
        "diameter": float(UOT_DIAMETER),
        "lambda_uot": float(FROZEN_LAMBDA_UOT),
        "full_seed_offset": int(FULL_TRAIN_SEED_OFFSET),
        "base_seeds": [int(x) for x in BASE_SEEDS],
    }


def seed_completion_signature(base_seed):
    d = frozen_config_dict().copy()
    d.update({
        "base_seed": int(base_seed),
        "actual_train_seed": int(base_seed) + int(FULL_TRAIN_SEED_OFFSET),
    })
    return d


def completion_signature_matches(path, base_seed):
    path = Path(path)
    if not path.exists():
        return False
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    expected = seed_completion_signature(base_seed)
    return all(saved.get(k) == v for k, v in expected.items())


# =====================================================================
# CONSOLE LOG
# =====================================================================
class Tee:
    def __init__(
        self,
        *streams,
    ):
        self.streams = streams

    def write(
        self,
        text,
    ):
        for stream in self.streams:
            stream.write(
                text
            )
            stream.flush()

    def flush(
        self,
    ):
        for stream in self.streams:
            stream.flush()


# =====================================================================
# HASH / COPY
# =====================================================================
def sha256_file(
    path,
    chunk_size=1024 * 1024,
):
    path = Path(
        path
    )

    h = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as f:
        while True:
            chunk = f.read(
                chunk_size
            )

            if not chunk:
                break

            h.update(
                chunk
            )

    return h.hexdigest()


def safe_copy(
    src,
    dst,
):
    src = Path(
        src
    )

    dst = Path(
        dst
    )

    dst.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        src,
        dst,
    )


def find_optional_previous_exp05_submission():
    """
    Previous final EXP05 test submission is optional.

    It is used ONLY to compare final test prediction behavior.
    """
    exact_candidates = [
        ROOT
        / "Experiments in order"
        / "0final_reproducible_pipeline"
        / "final_artifacts"
        / "submissions"
        / "FINAL_PRIMARY_EXP05_5SEED_AVERAGE.csv",

        ROOT
        / "final_artifacts"
        / "submissions"
        / "FINAL_PRIMARY_EXP05_5SEED_AVERAGE.csv",
    ]

    for p in exact_candidates:
        if p.exists():
            return p

    matches = list(
        ROOT.rglob(
            "FINAL_PRIMARY_EXP05_5SEED_AVERAGE.csv"
        )
    )

    if not matches:
        return None

    matches = sorted(
        matches,
        key=lambda p: (
            "0final_reproducible_pipeline"
            not in str(
                p
            ).lower(),
            len(
                p.parts
            ),
        ),
    )

    return matches[
        0
    ]


# =====================================================================
# DATASET WITH COUNTY INDEX
# =====================================================================
class IndexedFinalDataset(Dataset):
    def __init__(
        self,
        arrays,
        with_targets=True,
    ):
        self.base_dataset = (
            base.FinalCountyDataset(
                arrays,
                with_targets=with_targets,
            )
        )

        self.with_targets = (
            with_targets
        )

    def __len__(self):
        return len(
            self.base_dataset
        )

    def __getitem__(
        self,
        index,
    ):
        item = (
            self.base_dataset[
                index
            ]
        )

        if self.with_targets:
            (
                history,
                future,
                target,
                mask,
            ) = item

            return (
                history,
                future,
                target,
                mask,
                index,
            )

        (
            history,
            future,
        ) = item

        return (
            history,
            future,
            index,
        )


def state_codes_from_arrays(
    arrays,
):
    county_ids = np.asarray(
        arrays[
            "county_ids"
        ],
        dtype=np.int64,
    )

    states = (
        county_ids
        // 1000
    ).astype(
        np.int64
    )

    unknown = sorted(
        set(
            states.tolist()
        )
        - set(
            STATE_NAMES.keys()
        )
    )

    if unknown:
        raise ValueError(
            f"Unexpected state FIPS codes: {unknown}"
        )

    return states


# =====================================================================
# FROZEN STATE-UOT LOSS
# =====================================================================
_UOT_LOSS = None


def get_uot_loss():
    global _UOT_LOSS

    if _UOT_LOSS is None:
        _UOT_LOSS = SamplesLoss(
            loss="sinkhorn",
            p=2,
            blur=UOT_BLUR,
            reach=UOT_REACH,
            diameter=UOT_DIAMETER,
            debias=True,
            backend="tensorized",
        )

    return _UOT_LOSS


def categorical_support(
    n_counties,
    n_times,
    device,
    dtype,
):
    eye = torch.eye(
        n_counties,
        device=device,
        dtype=dtype,
    )

    return (
        eye.unsqueeze(0)
        .repeat(
            n_times,
            1,
            1,
        )
        .contiguous()
    )


def raw_severity_mass(
    x_log,
):
    return torch.clamp(
        torch.expm1(
            x_log
        ),
        min=0.0,
    )


def state_conditioned_uot_loss(
    prediction_log,
    target_log,
    mask,
    batch_state_codes,
):
    """
    Canonical unique physical-target representation:
        prediction[:, :-1, 0]
    gives the 143 unique future physical targets via the +1h route.

    UOT is solved independently within each state represented in the
    minibatch; state losses are weighted by county count / batch size.
    """
    pred = (
        prediction_log[
            :,
            :-1,
            0,
        ]
    )

    target = (
        target_log[
            :,
            :-1,
            0,
        ]
    )

    valid = (
        mask[
            :,
            :-1,
            0,
        ]
        > 0.5
    )

    if not bool(
        valid.all().item()
    ):
        raise RuntimeError(
            "Unexpected invalid canonical State-UOT cells."
        )

    pred_mass_all = (
        raw_severity_mass(
            pred
        )
    )

    target_mass_all = (
        raw_severity_mass(
            target
        )
    )

    batch_size = int(
        prediction_log.shape[0]
    )

    loss_fn = (
        get_uot_loss()
    )

    weighted = []

    for state_code in torch.unique(
        batch_state_codes,
        sorted=True,
    ):
        idx = (
            batch_state_codes
            == state_code
        )

        n_state = int(
            idx.sum().item()
        )

        pred_mass = (
            pred_mass_all[
                idx
            ]
            .transpose(
                0,
                1,
            )
            .contiguous()
        )

        target_mass = (
            target_mass_all[
                idx
            ]
            .transpose(
                0,
                1,
            )
            .contiguous()
        )

        pred_mass = (
            pred_mass
            + MASS_EPS
        ).contiguous()

        target_mass = (
            target_mass
            + MASS_EPS
        ).contiguous()

        n_times = int(
            target_mass.shape[0]
        )

        support = categorical_support(
            n_counties=n_state,
            n_times=n_times,
            device=prediction_log.device,
            dtype=prediction_log.dtype,
        )

        state_loss = (
            loss_fn(
                target_mass,
                support,
                pred_mass,
                support,
            )
            .mean()
        )

        weighted.append(
            (
                n_state
                / float(
                    batch_size
                )
            )
            * state_loss
        )

    if not weighted:
        return (
            prediction_log.sum()
            * 0.0
        )

    return torch.stack(
        weighted
    ).sum()


def lambda_for_epoch(
    epoch,
):
    if epoch <= WARMUP_EPOCHS:
        return 0.0

    ramp_end = (
        WARMUP_EPOCHS
        + RAMP_EPOCHS
    )

    if epoch <= ramp_end:
        fraction = (
            epoch
            - WARMUP_EPOCHS
        ) / float(
            RAMP_EPOCHS
        )

        return float(
            FROZEN_LAMBDA_UOT
            * fraction
        )

    return float(
        FROZEN_LAMBDA_UOT
    )


# =====================================================================
# FULL-DATA TRAINING
# =====================================================================
def train_one_full_seed(
    train_arrays,
    train_states,
    base_seed,
):
    actual_seed = (
        int(
            base_seed
        )
        + FULL_TRAIN_SEED_OFFSET
    )

    seed_dir = (
        MODEL_DIR
        / f"seed{base_seed}"
    )

    seed_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        seed_dir
        / f"FINAL_TRIAL22_STATE_UOT_seed{base_seed}_model.pt"
    )

    history_path = (
        seed_dir
        / f"FINAL_TRIAL22_STATE_UOT_seed{base_seed}_history.csv"
    )

    seed_submission_path = (
        SUBMISSION_DIR
        / f"FINAL_TRIAL22_STATE_UOT_seed{base_seed}.csv"
    )

    raw_prediction_path = (
        RAW_PRED_DIR
        / f"FINAL_TRIAL22_STATE_UOT_seed{base_seed}.npz"
    )

    completion_path = (
        seed_dir
        / "TRIAL22_COMPLETED.json"
    )

    complete = all(
        p.exists()
        for p in [
            checkpoint_path,
            history_path,
            seed_submission_path,
            raw_prediction_path,
            completion_path,
        ]
    )

    if complete and not completion_signature_matches(
        completion_path,
        base_seed,
    ):
        raise RuntimeError(
            f"Existing completed artifacts for base seed {base_seed} do not "
            "match the frozen Trial-22 configuration. Refusing to mix models. "
            f"Move or delete this seed directory before rerunning: {seed_dir}"
        )

    if (
        RESUME_COMPLETED_SEEDS
        and complete
    ):
        print(
            f"\n[RESUME] complete full-data base seed {base_seed}"
        )

        submission = pd.read_csv(
            seed_submission_path
        )

        discrepancy = (
            base.validate_submission(
                submission,
                f"Trial22 State-UOT seed {base_seed}",
            )
        )

        return {
            "base_seed": (
                base_seed
            ),
            "actual_seed": (
                actual_seed
            ),
            "submission": (
                submission
            ),
            "submission_path": (
                seed_submission_path
            ),
            "history_path": (
                history_path
            ),
            "checkpoint_path": (
                checkpoint_path
            ),
            "raw_prediction_path": (
                raw_prediction_path
            ),
            "completion_path": (
                completion_path
            ),
            "target_discrepancy": (
                discrepancy
            ),
            "reused": True,
        }

    print("\n" + "-" * 78)

    print(
        f"FULL-DATA TRIAL22 STATE-UOT | base_seed={base_seed} | "
        f"actual_train_seed={actual_seed}"
    )

    print(
        f"reach={UOT_REACH:.2f} | "
        f"blur={UOT_BLUR:.2f} | "
        f"lambda={FROZEN_LAMBDA_UOT:.15g}"
    )

    base.set_seed(
        actual_seed
    )

    dataset = IndexedFinalDataset(
        train_arrays,
        with_targets=True,
    )

    generator = torch.Generator()

    generator.manual_seed(
        actual_seed
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(
            base.DEVICE.type
            == "cuda"
        ),
        generator=generator,
    )

    model = (
        base.FinalCNNV2()
        .to(
            base.DEVICE
        )
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
    )

    rows = []

    started = time.time()

    for epoch in range(
        1,
        EPOCHS + 1,
    ):
        current_lambda = (
            lambda_for_epoch(
                epoch
            )
        )

        model.train()

        running_point = 0.0
        running_uot = 0.0
        running_total = 0.0
        total_valid = 0.0
        n_batches = 0

        for (
            history,
            future,
            target,
            mask,
            index,
        ) in loader:
            history = history.to(
                base.DEVICE,
                non_blocking=True,
            )

            future = future.to(
                base.DEVICE,
                non_blocking=True,
            )

            target = target.to(
                base.DEVICE,
                non_blocking=True,
            )

            mask = mask.to(
                base.DEVICE,
                non_blocking=True,
            )

            batch_states = (
                train_states[
                    index
                ]
                .to(
                    base.DEVICE,
                    non_blocking=True,
                )
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            pred = model(
                history,
                future,
            )

            point = (
                base.masked_smooth_l1(
                    pred,
                    target,
                    mask,
                )
            )

            uot = (
                state_conditioned_uot_loss(
                    pred,
                    target,
                    mask,
                    batch_states,
                )
            )

            total = (
                point
                + current_lambda
                * uot
            )

            total.backward()
            optimizer.step()

            n_valid = float(
                mask.sum().item()
            )

            running_point += (
                float(
                    point.detach()
                    .cpu()
                    .item()
                )
                * n_valid
            )

            total_valid += (
                n_valid
            )

            running_uot += float(
                uot.detach()
                .cpu()
                .item()
            )

            running_total += float(
                total.detach()
                .cpu()
                .item()
            )

            n_batches += 1

        point_epoch = (
            running_point
            / max(
                total_valid,
                1.0,
            )
        )

        uot_epoch = (
            running_uot
            / max(
                n_batches,
                1,
            )
        )

        total_epoch = (
            running_total
            / max(
                n_batches,
                1,
            )
        )

        rows.append({
            "base_seed": (
                base_seed
            ),
            "actual_train_seed": (
                actual_seed
            ),
            "epoch": (
                epoch
            ),
            "point_loss": (
                point_epoch
            ),
            "state_uot_loss": (
                uot_epoch
            ),
            "lambda_uot": (
                current_lambda
            ),
            "lambda_times_uot": (
                current_lambda
                * uot_epoch
            ),
            "total_loss": (
                total_epoch
            ),
        })

        if (
            epoch == 1
            or epoch % 20 == 0
            or epoch in [
                WARMUP_EPOCHS,
                WARMUP_EPOCHS + 1,
                WARMUP_EPOCHS + RAMP_EPOCHS,
            ]
            or epoch == EPOCHS
        ):
            print(
                f"  epoch {epoch:3d}/{EPOCHS}: "
                f"point={point_epoch:.8f} "
                f"stateUOT={uot_epoch:.8f} "
                f"lambda={current_lambda:.8g}"
            )

    history = pd.DataFrame(
        rows
    )

    history.to_csv(
        history_path,
        index=False,
    )

    elapsed = (
        time.time()
        - started
    )

    print(
        f"  training completed in "
        f"{elapsed/60.0:.2f} min"
    )

    return {
        "base_seed": (
            base_seed
        ),
        "actual_seed": (
            actual_seed
        ),
        "model": (
            model
        ),
        "history": (
            history
        ),
        "history_path": (
            history_path
        ),
        "checkpoint_path": (
            checkpoint_path
        ),
        "seed_submission_path": (
            seed_submission_path
        ),
        "raw_prediction_path": (
            raw_prediction_path
        ),
        "completion_path": (
            completion_path
        ),
        "elapsed_seconds": (
            elapsed
        ),
        "reused": False,
    }


# =====================================================================
# SUBMISSION / ENSEMBLE
# =====================================================================
def make_seed_submission(
    seed_result,
    test_arrays,
):
    if seed_result[
        "reused"
    ]:
        return (
            seed_result[
                "submission"
            ],
            seed_result[
                "target_discrepancy"
            ],
        )

    model = seed_result[
        "model"
    ]

    prediction = (
        base.final_predict(
            model,
            test_arrays,
        )
    )

    np.savez_compressed(
        seed_result[
            "raw_prediction_path"
        ],
        prediction=prediction,
        county_ids=np.asarray(
            test_arrays[
                "county_ids"
            ]
        ),
        timestamps=np.asarray(
            test_arrays[
                "timestamps"
            ]
        ),
        horizons=np.asarray(
            base.HORIZONS
        ),
    )

    submission = (
        base.build_submission_from_prediction(
            test_arrays,
            prediction,
        )
    )

    discrepancy = (
        base.validate_submission(
            submission,
            (
                f"full-data Trial22 State-UOT "
                f"base seed {seed_result['base_seed']}"
            ),
        )
    )

    submission.to_csv(
        seed_result[
            "seed_submission_path"
        ],
        index=False,
    )

    torch.save(
        {
            "state_dict": (
                model.state_dict()
            ),
            "method": (
                "trial22_state_conditioned_uot"
            ),
            "architecture": (
                "FinalCNNV2"
            ),
            "base_seed": (
                seed_result[
                    "base_seed"
                ]
            ),
            "actual_train_seed": (
                seed_result[
                    "actual_seed"
                ]
            ),
            "reach": UOT_REACH,
            "blur": UOT_BLUR,
            "diameter": UOT_DIAMETER,
            "lambda_uot": (
                FROZEN_LAMBDA_UOT
            ),
            "warmup_epochs": (
                WARMUP_EPOCHS
            ),
            "ramp_epochs": (
                RAMP_EPOCHS
            ),
            "epochs": EPOCHS,
            "batch_size": (
                BATCH_SIZE
            ),
            "learning_rate": (
                LR
            ),
            "history_features": (
                list(
                    base.HISTORY_COLS
                )
            ),
            "future_features": (
                list(
                    base.FUTURE_COLS
                )
            ),
            "horizons": (
                list(
                    base.HORIZONS
                )
            ),
        },
        seed_result[
            "checkpoint_path"
        ],
    )

    # Mark the seed complete only AFTER checkpoint, history, raw predictions,
    # validated seed submission, and frozen metadata have all been written.
    seed_result[
        "completion_path"
    ].write_text(
        json.dumps(
            seed_completion_signature(
                seed_result["base_seed"]
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return (
        submission,
        discrepancy,
    )


def average_seed_submissions(
    seed_submissions,
):
    template = pd.read_csv(
        base.TEMPLATE_PATH
    )

    final = template.copy()

    official_nan = (
        template[
            base.TARGET_COLS
        ]
        .isna()
        .to_numpy()
    )

    for col in base.TARGET_COLS:
        matrix = np.column_stack(
            [
                s[
                    col
                ].to_numpy(float)
                for s in seed_submissions
            ]
        )

        averaged = np.nanmean(
            matrix,
            axis=1,
        )

        all_nan = np.all(
            ~np.isfinite(
                matrix
            ),
            axis=1,
        )

        averaged[
            all_nan
        ] = np.nan

        final[
            col
        ] = averaged

    final[
        base.TARGET_COLS
    ] = (
        final[
            base.TARGET_COLS
        ]
        .mask(
            official_nan
        )
    )

    discrepancy = (
        base.validate_submission(
            final,
            "FINAL_TRIAL22_STATE_UOT_5SEED_AVERAGE",
        )
    )

    return (
        final,
        discrepancy,
    )


# =====================================================================
# TEST DIAGNOSTICS
# =====================================================================
def seed_disagreement_table(
    seed_submissions,
):
    template = pd.read_csv(
        base.TEMPLATE_PATH
    )

    rows = template[
        [
            "fipsCode",
            "countyName",
            "stateAbbr",
            "timestamp_et",
        ]
    ].copy()

    for col in base.TARGET_COLS:
        matrix = np.column_stack(
            [
                s[
                    col
                ].to_numpy(float)
                for s in seed_submissions
            ]
        )

        rows[
            f"{col}__mean"
        ] = np.nanmean(
            matrix,
            axis=1,
        )

        rows[
            f"{col}__std"
        ] = np.nanstd(
            matrix,
            axis=1,
            ddof=1,
        )

        rows[
            f"{col}__min"
        ] = np.nanmin(
            matrix,
            axis=1,
        )

        rows[
            f"{col}__max"
        ] = np.nanmax(
            matrix,
            axis=1,
        )

    return rows


def seed_disagreement_summary(
    disagreement,
):
    rows = []

    for col in base.TARGET_COLS:
        values = (
            disagreement[
                f"{col}__std"
            ]
            .to_numpy(float)
        )

        values = values[
            np.isfinite(
                values
            )
        ]

        rows.append({
            "target_column": (
                col
            ),
            "mean_seed_std": float(
                np.mean(
                    values
                )
            ),
            "median_seed_std": float(
                np.median(
                    values
                )
            ),
            "p90_seed_std": float(
                np.quantile(
                    values,
                    0.90,
                )
            ),
            "p95_seed_std": float(
                np.quantile(
                    values,
                    0.95,
                )
            ),
            "p99_seed_std": float(
                np.quantile(
                    values,
                    0.99,
                )
            ),
            "max_seed_std": float(
                np.max(
                    values
                )
            ),
        })

    return pd.DataFrame(
        rows
    )


def prediction_distribution_summary(
    final_submission,
):
    rows = []

    for col in base.TARGET_COLS:
        values = (
            final_submission[
                col
            ]
            .to_numpy(float)
        )

        values = values[
            np.isfinite(
                values
            )
        ]

        rows.append({
            "target_column": (
                col
            ),
            "n_finite": len(
                values
            ),
            "mean": float(
                np.mean(
                    values
                )
            ),
            "std": float(
                np.std(
                    values,
                    ddof=1,
                )
            ),
            "min": float(
                np.min(
                    values
                )
            ),
            "q50": float(
                np.quantile(
                    values,
                    0.50,
                )
            ),
            "q75": float(
                np.quantile(
                    values,
                    0.75,
                )
            ),
            "q90": float(
                np.quantile(
                    values,
                    0.90,
                )
            ),
            "q95": float(
                np.quantile(
                    values,
                    0.95,
                )
            ),
            "q99": float(
                np.quantile(
                    values,
                    0.99,
                )
            ),
            "max": float(
                np.max(
                    values
                )
            ),
            "sum": float(
                np.sum(
                    values
                )
            ),
        })

    return pd.DataFrame(
        rows
    )


def compare_to_exp05_submission(
    state_submission,
):
    path = (
        find_optional_previous_exp05_submission()
    )

    if path is None:
        print(
            "\n[INFO] Previous final EXP05 test submission not found. "
            "Skipping test-prediction comparison."
        )

        return (
            None,
            None,
        )

    print(
        f"\nPrevious EXP05 test submission found:\n  {path}"
    )

    exp05 = pd.read_csv(
        path
    )

    base.validate_submission(
        exp05,
        "previous final EXP05 test submission",
    )

    rows = []

    for col in base.TARGET_COLS:
        a = exp05[
            col
        ].to_numpy(float)

        b = state_submission[
            col
        ].to_numpy(float)

        mask = (
            np.isfinite(
                a
            )
            & np.isfinite(
                b
            )
        )

        aa = a[
            mask
        ]

        bb = b[
            mask
        ]

        diff = (
            bb-aa
        )

        rows.append({
            "target_column": (
                col
            ),
            "n": len(
                diff
            ),
            "mean_STATE_minus_EXP05": float(
                np.mean(
                    diff
                )
            ),
            "MAE_between_models": float(
                np.mean(
                    np.abs(
                        diff
                    )
                )
            ),
            "RMSE_between_models": float(
                np.sqrt(
                    np.mean(
                        diff**2
                    )
                )
            ),
            "prediction_correlation": float(
                np.corrcoef(
                    aa,
                    bb,
                )[
                    0,
                    1,
                ]
            ),
            "EXP05_mean": float(
                np.mean(
                    aa
                )
            ),
            "STATE_UOT_mean": float(
                np.mean(
                    bb
                )
            ),
        })

    comparison = pd.DataFrame(
        rows
    )

    safe_copy(
        path,
        PROVENANCE_DIR
        / "PREVIOUS_FINAL_EXP05_5SEED_TEST_SUBMISSION.csv",
    )

    comparison.to_csv(
        METRICS_DIR
        / "STATE_UOT_vs_EXP05_test_prediction_comparison.csv",
        index=False,
    )

    return (
        exp05,
        comparison,
    )


# =====================================================================
# VISUALIZATIONS
# =====================================================================
def save_training_visualizations(
    histories,
):
    combined = pd.concat(
        histories,
        ignore_index=True,
    )

    combined.to_csv(
        METRICS_DIR
        / "all_full_data_training_histories.csv",
        index=False,
    )

    mean_history = (
        combined.groupby(
            "epoch",
            as_index=False,
        )
        .agg(
            point_loss_mean=(
                "point_loss",
                "mean",
            ),
            point_loss_std=(
                "point_loss",
                "std",
            ),
            state_uot_loss_mean=(
                "state_uot_loss",
                "mean",
            ),
            state_uot_loss_std=(
                "state_uot_loss",
                "std",
            ),
            lambda_uot=(
                "lambda_uot",
                "mean",
            ),
            weighted_uot_mean=(
                "lambda_times_uot",
                "mean",
            ),
        )
    )

    mean_history.to_csv(
        METRICS_DIR
        / "full_data_training_history_mean_across_5_seeds.csv",
        index=False,
    )

    plt.figure(
        figsize=(9, 5)
    )

    for seed, d in combined.groupby(
        "base_seed"
    ):
        plt.plot(
            d[
                "epoch"
            ],
            d[
                "point_loss"
            ],
            alpha=0.55,
            label=f"seed {seed}",
        )

    plt.xlabel(
        "Epoch"
    )
    plt.ylabel(
        "Point loss"
    )
    plt.title(
        "Final Trial-22 State-UOT full-data training: point loss"
    )
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIG_DIR
        / "01_training_point_loss_by_seed.png",
        dpi=200,
    )
    plt.close()

    plt.figure(
        figsize=(9, 5)
    )

    for seed, d in combined.groupby(
        "base_seed"
    ):
        plt.plot(
            d[
                "epoch"
            ],
            d[
                "state_uot_loss"
            ],
            alpha=0.55,
            label=f"seed {seed}",
        )

    plt.xlabel(
        "Epoch"
    )
    plt.ylabel(
        "State-UOT loss"
    )
    plt.title(
        "Final Trial-22 State-UOT full-data training: distribution loss"
    )
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIG_DIR
        / "02_training_state_uot_loss_by_seed.png",
        dpi=200,
    )
    plt.close()

    plt.figure(
        figsize=(9, 5)
    )

    plt.plot(
        mean_history[
            "epoch"
        ],
        mean_history[
            "lambda_uot"
        ],
    )

    plt.xlabel(
        "Epoch"
    )
    plt.ylabel(
        "UOT lambda"
    )
    plt.title(
        "Final frozen Trial-22 State-UOT warm-up / ramp"
    )
    plt.tight_layout()

    plt.savefig(
        FIG_DIR
        / "03_lambda_schedule.png",
        dpi=200,
    )
    plt.close()


def save_test_visualizations(
    final_submission,
    disagreement_summary,
    exp05_submission=None,
):
    # Prediction quantiles by horizon.
    labels = []
    q50 = []
    q90 = []
    q95 = []
    q99 = []

    for col in base.TARGET_COLS:
        v = (
            final_submission[
                col
            ]
            .to_numpy(float)
        )

        v = v[
            np.isfinite(
                v
            )
        ]

        labels.append(
            col.replace(
                "osi_target_t",
                "",
            )
        )

        q50.append(
            np.quantile(
                v,
                0.50,
            )
        )

        q90.append(
            np.quantile(
                v,
                0.90,
            )
        )

        q95.append(
            np.quantile(
                v,
                0.95,
            )
        )

        q99.append(
            np.quantile(
                v,
                0.99,
            )
        )

    plt.figure(
        figsize=(9, 5)
    )

    plt.plot(
        labels,
        q50,
        marker="o",
        label="Q50",
    )

    plt.plot(
        labels,
        q90,
        marker="o",
        label="Q90",
    )

    plt.plot(
        labels,
        q95,
        marker="o",
        label="Q95",
    )

    plt.plot(
        labels,
        q99,
        marker="o",
        label="Q99",
    )

    plt.xlabel(
        "Forecast horizon"
    )
    plt.ylabel(
        "Predicted OSI"
    )
    plt.title(
        "Final Trial-22 State-UOT test prediction quantiles"
    )
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIG_DIR
        / "04_test_prediction_quantiles.png",
        dpi=200,
    )
    plt.close()

    # Seed disagreement.
    plt.figure(
        figsize=(9, 5)
    )

    x = np.arange(
        len(
            disagreement_summary
        )
    )

    plt.plot(
        x,
        disagreement_summary[
            "mean_seed_std"
        ],
        marker="o",
        label="Mean seed SD",
    )

    plt.plot(
        x,
        disagreement_summary[
            "p90_seed_std"
        ],
        marker="o",
        label="P90 seed SD",
    )

    plt.plot(
        x,
        disagreement_summary[
            "p99_seed_std"
        ],
        marker="o",
        label="P99 seed SD",
    )

    plt.xticks(
        x,
        [
            c.replace(
                "osi_target_t",
                "",
            )
            for c in disagreement_summary[
                "target_column"
            ]
        ],
    )

    plt.xlabel(
        "Forecast horizon"
    )
    plt.ylabel(
        "Across-seed prediction SD"
    )
    plt.title(
        "Final test prediction disagreement across five seeds"
    )
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIG_DIR
        / "05_test_seed_disagreement.png",
        dpi=200,
    )
    plt.close()

    if exp05_submission is not None:
        # Model scatter, one figure per horizon.
        for col in base.TARGET_COLS:
            a = (
                exp05_submission[
                    col
                ].to_numpy(float)
            )

            b = (
                final_submission[
                    col
                ].to_numpy(float)
            )

            mask = (
                np.isfinite(
                    a
                )
                & np.isfinite(
                    b
                )
            )

            aa = a[
                mask
            ]

            bb = b[
                mask
            ]

            vmax = float(
                max(
                    np.max(
                        aa
                    ),
                    np.max(
                        bb
                    ),
                )
            )

            plt.figure(
                figsize=(7, 7)
            )

            plt.scatter(
                aa,
                bb,
                alpha=0.25,
                s=12,
            )

            plt.plot(
                [
                    0,
                    vmax,
                ],
                [
                    0,
                    vmax,
                ],
                linestyle="--",
                linewidth=1,
            )

            plt.xlabel(
                "EXP05 final test prediction"
            )

            plt.ylabel(
                "Trial-22 State-UOT final test prediction"
            )

            plt.title(
                f"Final test predictions: "
                f"{col.replace('osi_target_t', '')}"
            )

            plt.tight_layout()

            plt.savefig(
                FIG_DIR
                / (
                    "06_compare_EXP05_STATE_"
                    + col
                    + ".png"
                ),
                dpi=200,
            )

            plt.close()


def build_unique_physical_test_table(
    final_submission,
):
    """
    Collapse official origin/horizon routes to physical target time.
    The final submission is validated to have zero route discrepancy.
    """
    parts = []

    origin = pd.to_datetime(
        final_submission[
            "timestamp_et"
        ]
    )

    for col, h in zip(
        base.TARGET_COLS,
        base.HORIZONS,
    ):
        z = final_submission[
            [
                "fipsCode",
                "countyName",
                "stateAbbr",
            ]
        ].copy()

        z[
            "target_timestamp"
        ] = (
            origin
            + pd.to_timedelta(
                int(
                    h
                ),
                unit="h",
            )
        )

        z[
            "predicted_osi"
        ] = (
            final_submission[
                col
            ]
        )

        z = z[
            z[
                "predicted_osi"
            ].notna()
        ]

        parts.append(
            z
        )

    long = pd.concat(
        parts,
        ignore_index=True,
    )

    unique = (
        long.groupby(
            [
                "fipsCode",
                "countyName",
                "stateAbbr",
                "target_timestamp",
            ],
            as_index=False,
        )
        .agg(
            predicted_osi=(
                "predicted_osi",
                "mean",
            ),
            route_min=(
                "predicted_osi",
                "min",
            ),
            route_max=(
                "predicted_osi",
                "max",
            ),
            n_routes=(
                "predicted_osi",
                "size",
            ),
        )
    )

    unique[
        "route_discrepancy"
    ] = (
        unique[
            "route_max"
        ]
        - unique[
            "route_min"
        ]
    )

    return unique


def save_top_peak_trajectories(
    final_submission,
):
    unique = (
        build_unique_physical_test_table(
            final_submission
        )
    )

    unique.to_csv(
        METRICS_DIR
        / "final_test_unique_physical_targets.csv",
        index=False,
    )

    top_counties = (
        unique.groupby(
            "fipsCode"
        )[
            "predicted_osi"
        ]
        .max()
        .sort_values(
            ascending=False
        )
        .head(8)
        .index
        .tolist()
    )

    for rank, fips in enumerate(
        top_counties,
        start=1,
    ):
        d = unique[
            unique[
                "fipsCode"
            ]
            == fips
        ].sort_values(
            "target_timestamp"
        )

        county = (
            d[
                "countyName"
            ].iloc[0]
        )

        state = (
            d[
                "stateAbbr"
            ].iloc[0]
        )

        plt.figure(
            figsize=(11, 5)
        )

        plt.plot(
            d[
                "target_timestamp"
            ],
            d[
                "predicted_osi"
            ],
            linewidth=2,
        )

        plt.xlabel(
            "Physical target time"
        )
        plt.ylabel(
            "Predicted OSI"
        )
        plt.title(
            f"Final Trial-22 State-UOT predicted trajectory - "
            f"{county}, {state} ({int(fips)})"
        )
        plt.xticks(
            rotation=30
        )
        plt.tight_layout()

        plt.savefig(
            FIG_DIR
            / (
                f"07_peak_{rank:02d}_"
                f"county_{int(fips)}.png"
            ),
            dpi=200,
        )

        plt.close()


# =====================================================================
# PROVENANCE / ARTIFACT INVENTORY
# =====================================================================
def save_provenance(
    previous_winner_dir,
    phase11_dir,
):
    source_rows = []

    targets = [
        (
            "DM_Train.csv",
            Path(
                base.TRAIN_PATH
            ),
        ),
        (
            "DM_Test.csv",
            Path(
                base.TEST_PATH
            ),
        ),
        (
            "sample_submission.csv",
            Path(
                base.TEMPLATE_PATH
            ),
        ),
        (
            "FINAL_TRIAL22_END_TO_END.py",
            Path(
                __file__
            ),
        ),
    ]

    for label, path in targets:
        source_rows.append({
            "label": label,
            "path": str(
                path
            ),
            "exists": (
                path.exists()
            ),
            "size_bytes": (
                int(
                    path.stat().st_size
                )
                if path.exists()
                else np.nan
            ),
            "sha256": (
                sha256_file(
                    path
                )
                if path.exists()
                else ""
            ),
        })

    pd.DataFrame(
        source_rows
    ).to_csv(
        PROVENANCE_DIR
        / "input_source_sha256_manifest.csv",
        index=False,
    )

    safe_copy(
        Path(
            __file__
        ),
        PROVENANCE_DIR
        / "phase14_trial22_source_snapshot.py",
    )

    (
        PROVENANCE_DIR
        / "embedded_final_reproducible_pipeline_source.py"
    ).write_text(
        EMBEDDED_BASE_SOURCE,
        encoding="utf-8",
    )

    # Preserve model-selection decisions if present.
    if previous_winner_dir.exists():
        for name in [
            "FINAL_WINNER_STATUS.json",
            "MASTER_FINAL_MODEL_SUMMARY.csv",
            "MASTER_FINAL_MODEL_HORIZON_COMPARISON.csv",
        ]:
            src = (
                previous_winner_dir
                / name
            )

            if src.exists():
                safe_copy(
                    src,
                    PROVENANCE_DIR
                    / (
                        "PRE_PHASE11_"
                        + name
                    ),
                )

    if phase11_dir.exists():
        for name in [
            "PHASE11_DECISION.json",
            "PREDECLARED_GUARDRAIL_RESULTS.csv",
            "MASTER_PHASE11_BLEND_SELECTION_TABLE.csv",
        ]:
            src = (
                phase11_dir
                / name
            )

            if src.exists():
                safe_copy(
                    src,
                    PROVENANCE_DIR
                    / name,
                )

    # Preserve the hyperparameter-search and final robustness-confirmation
    # records when they are present. These are documentation only and are
    # never read by the training/prediction path.
    stage1_dir = ROOT / "stage1_state_uot_tuning"
    if stage1_dir.exists():
        for name in [
            "stage1_leaderboard.csv",
            "stage1_trials.csv",
            "run_config.json",
            "baseline_trial0.json",
        ]:
            src = stage1_dir / name
            if src.exists():
                safe_copy(
                    src,
                    PROVENANCE_DIR
                    / ("STAGE1_" + name),
                )

    confirmation_dir = (
        ROOT
        / "trial22_final_confirmation_5seed_5fold"
        / "final_summary"
    )
    if confirmation_dir.exists():
        for name in [
            "TRIAL22_vs_INCUMBENT_5SEED_ENSEMBLE.csv",
            "PER_SEED_TRIAL22_vs_INCUMBENT.csv",
            "COUNTY_BOOTSTRAP_HORIZON_COMPARISON.csv",
            "COUNTY_BOOTSTRAP_OVERALL_OBJECTIVE.csv",
            "FINAL_CONFIRMATION_SUMMARY.json",
        ]:
            src = confirmation_dir / name
            if src.exists():
                safe_copy(
                    src,
                    PROVENANCE_DIR
                    / ("PHASE13_" + name),
                )

    environment = {
        "generated_at_local": (
            datetime.now().isoformat()
        ),
        "python": sys.version,
        "platform": (
            platform.platform()
        ),
        "numpy": (
            np.__version__
        ),
        "pandas": (
            pd.__version__
        ),
        "torch": (
            torch.__version__
        ),
        "torch_cuda_version": (
            torch.version.cuda
        ),
        "cuda_available": (
            torch.cuda.is_available()
        ),
        "gpu_name": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
        "geomloss": (
            getattr(
                geomloss,
                "__version__",
                "unknown",
            )
        ),
        "device": str(
            base.DEVICE
        ),
        "output_folder": str(
            OUT
        ),
    }

    (
        PROVENANCE_DIR
        / "environment.json"
    ).write_text(
        json.dumps(
            environment,
            indent=2,
        ),
        encoding="utf-8",
    )


def write_artifact_manifest():
    manifest_path = (
        OUT
        / "ARTIFACT_MANIFEST.csv"
    )

    rows = []

    for path in sorted(
        OUT.rglob(
            "*"
        )
    ):
        if (
            not path.is_file()
            or path
            == manifest_path
            or path.name
            == "PHASE14_TRIAL22_CONSOLE.txt"
        ):
            continue

        rows.append({
            "relative_path": str(
                path.relative_to(
                    OUT
                )
            ),
            "size_bytes": int(
                path.stat().st_size
            ),
            "sha256": (
                sha256_file(
                    path
                )
            ),
        })

    pd.DataFrame(
        rows
    ).to_csv(
        manifest_path,
        index=False,
    )



# =====================================================================
# COMPETITION-READY MATERIAL
# =====================================================================

def build_competition_ready_material(final_submission, final_path):
    """Create the prediction file and reproducible code.zip required by the challenge."""
    competition_prediction = (
        COMPETITION_DIR
        / "prediction_file.csv"
    )

    # Write from the already validated DataFrame, not by manual editing.
    final_submission.to_csv(
        competition_prediction,
        index=False,
    )

    # Revalidate the exact competition copy.
    check = pd.read_csv(
        competition_prediction
    )
    competition_discrepancy = (
        base.validate_submission(
            check,
            "competition_ready/prediction_file.csv",
        )
    )

    requirements = """numpy==2.0.2
pandas==2.2.2
scikit-learn==1.6.1
scipy==1.13.1
torch==2.6.0
geomloss==0.3.1
matplotlib
"""

    requirements_path = (
        COMPETITION_DIR
        / "requirements.txt"
    )
    requirements_path.write_text(
        requirements,
        encoding="utf-8",
    )

    code_readme = f"""INFORMS 2026 DATA MINING SOCIETY DATA CHALLENGE
REPRODUCIBLE FINAL CODE - TRIAL-22 STATE-UOT
================================================

Files in code.zip
-----------------
1. FINAL_TRIAL22_END_TO_END.py
2. requirements.txt
3. README_CODE.txt

How to reproduce
----------------
Place the following official data files in the SAME directory as FINAL_TRIAL22_END_TO_END.py:
    DM_Train.csv
    DM_Test.csv
    sample_submission.csv

Install the dependencies in requirements.txt, then run:
    python FINAL_TRIAL22_END_TO_END.py

The script trains five full-data models with actual seeds:
    {[x + FULL_TRAIN_SEED_OFFSET for x in BASE_SEEDS]}

It creates the five-seed averaged prediction and validates it against the official
template, including identifier/order preservation, required NaNs, non-negativity,
and physical-target consistency.

Frozen Trial-22 hyperparameters
-------------------------------
reach={UOT_REACH}
blur={UOT_BLUR}
diameter={UOT_DIAMETER}
lambda_uot={FROZEN_LAMBDA_UOT}
warmup_epochs={WARMUP_EPOCHS}
ramp_epochs={RAMP_EPOCHS}
epochs={EPOCHS}
batch_size={BATCH_SIZE}
learning_rate={LR}

Expected competition prediction:
    final_trial22_state_uot_submission/competition_ready/prediction_file.csv

The written competition report is prepared separately and is not generated by this code.
"""

    code_readme_path = (
        COMPETITION_DIR
        / "README_CODE.txt"
    )
    code_readme_path.write_text(
        code_readme,
        encoding="utf-8",
    )

    code_zip_path = (
        COMPETITION_DIR
        / "code.zip"
    )

    with zipfile.ZipFile(
        code_zip_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as zf:
        zf.write(
            Path(__file__).resolve(),
            arcname="FINAL_TRIAL22_END_TO_END.py",
        )
        zf.write(
            requirements_path,
            arcname="requirements.txt",
        )
        zf.write(
            code_readme_path,
            arcname="README_CODE.txt",
        )

    material_status = {
        "prediction_file": str(competition_prediction),
        "prediction_file_sha256": sha256_file(competition_prediction),
        "source_final_submission": str(final_path),
        "code_zip": str(code_zip_path),
        "code_zip_sha256": sha256_file(code_zip_path),
        "competition_prediction_target_consistency": float(
            competition_discrepancy
        ),
        "written_report": (
            "REQUIRED SEPARATELY: PDF, no more than 6 pages excluding references"
        ),
    }

    (
        COMPETITION_DIR
        / "COMPETITION_MATERIAL_STATUS.json"
    ).write_text(
        json.dumps(
            material_status,
            indent=2,
        ),
        encoding="utf-8",
    )

    return (
        competition_prediction,
        code_zip_path,
        competition_discrepancy,
    )


# =====================================================================
# MAIN
# =====================================================================
def run():
    started = time.time()

    # Match the deterministic settings used during model selection/confirmation.
    base.configure_determinism()

    print("=" * 78)
    print(
        "PHASE 14 -- FINAL FULL-DATA TRIAL-22 STATE-UOT TRAINING AND SUBMISSION"
    )
    print("=" * 78)

    print(
        "Final selected model: FROZEN TRIAL-22 STATE-UOT."
    )

    print(
        "Trial 22 is frozen; no further tuning or blending will be used."
    )

    print(
        f"Device: {base.DEVICE}"
    )

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(
                0
            ),
        )

    print(
        "GeomLoss:",
        getattr(
            geomloss,
            "__version__",
            "unknown",
        ),
    )

    print(
        "\nFrozen Trial-22 State-UOT:"
    )
    print(
        f"  reach     : {UOT_REACH}"
    )
    print(
        f"  blur      : {UOT_BLUR}"
    )
    print(
        f"  diameter  : {UOT_DIAMETER:.12g}"
    )
    print(
        f"  lambda    : {FROZEN_LAMBDA_UOT:.15g}"
    )
    print(
        f"  warm-up   : {WARMUP_EPOCHS}"
    )
    print(
        f"  ramp      : {RAMP_EPOCHS}"
    )
    print(
        f"  epochs    : {EPOCHS}"
    )
    print(
        f"  seeds     : {BASE_SEEDS}"
    )
    print(
        f"  full seed : base_seed + {FULL_TRAIN_SEED_OFFSET}"
    )

    previous_winner_dir = (
        ROOT
        / "uot_experiments"
        / "The final model until this point"
    )

    phase11_dir = (
        ROOT
        / "ensemble_experiments"
        / "phase11_crossfitted_exp05_stateuot_blend_FINAL"
    )

    save_provenance(
        previous_winner_dir,
        phase11_dir,
    )

    # -------------------------------------------------------------
    # Data / full-data scalers
    # -------------------------------------------------------------
    train, test = (
        base.load_data()
    )

    train_ids = np.sort(
        train[
            "fipsCode"
        ]
        .astype(int)
        .unique()
    )

    test_ids = np.sort(
        test[
            "fipsCode"
        ]
        .astype(int)
        .unique()
    )

    print(
        f"\nTraining counties: {len(train_ids)}"
    )
    print(
        f"Test counties    : {len(test_ids)}"
    )

    (
        history_scaler,
        future_scaler,
    ) = (
        base.fit_final_scalers(
            train,
            train_ids,
        )
    )

    scaler_path = (
        MODEL_DIR
        / "FINAL_STATE_UOT_scalers.pkl"
    )

    with open(
        scaler_path,
        "wb",
    ) as f:
        pickle.dump(
            {
                "history_scaler": (
                    history_scaler
                ),
                "future_scaler": (
                    future_scaler
                ),
                "history_features": (
                    list(
                        base.HISTORY_COLS
                    )
                ),
                "future_features": (
                    list(
                        base.FUTURE_COLS
                    )
                ),
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    train_arrays = (
        base.build_final_county_arrays(
            train,
            train_ids,
            history_scaler,
            future_scaler,
            with_targets=True,
        )
    )

    test_arrays = (
        base.build_final_county_arrays(
            test,
            test_ids,
            history_scaler,
            future_scaler,
            with_targets=False,
        )
    )

    train_states = torch.from_numpy(
        state_codes_from_arrays(
            train_arrays
        )
    ).long()

    # Save full-data county inventory.
    pd.DataFrame({
        "fipsCode": (
            train_ids
        ),
        "state_fips": (
            train_ids
            // 1000
        ),
        "state": [
            STATE_NAMES[
                int(
                    x // 1000
                )
            ]
            for x in train_ids
        ],
    }).to_csv(
        PROVENANCE_DIR
        / "full_training_counties.csv",
        index=False,
    )

    # -------------------------------------------------------------
    # Five full-data seeds
    # -------------------------------------------------------------
    seed_submissions = []
    histories = []
    seed_validation_rows = []

    for base_seed in BASE_SEEDS:
        result = (
            train_one_full_seed(
                train_arrays,
                train_states,
                base_seed,
            )
        )

        submission, discrepancy = (
            make_seed_submission(
                result,
                test_arrays,
            )
        )

        seed_submissions.append(
            submission
        )

        history = pd.read_csv(
            result[
                "history_path"
            ]
        )

        histories.append(
            history
        )

        seed_validation_rows.append({
            "base_seed": (
                base_seed
            ),
            "actual_train_seed": (
                base_seed
                + FULL_TRAIN_SEED_OFFSET
            ),
            "max_duplicate_target_discrepancy": (
                discrepancy
            ),
            "reused": (
                result[
                    "reused"
                ]
            ),
        })

    seed_validation = pd.DataFrame(
        seed_validation_rows
    )

    seed_validation.to_csv(
        METRICS_DIR
        / "seed_submission_validation.csv",
        index=False,
    )

    # -------------------------------------------------------------
    # Five-seed final ensemble
    # -------------------------------------------------------------
    (
        final_submission,
        ensemble_discrepancy,
    ) = (
        average_seed_submissions(
            seed_submissions
        )
    )

    final_path = (
        SUBMISSION_DIR
        / FINAL_SUBMISSION_NAME
    )

    final_submission.to_csv(
        final_path,
        index=False,
    )

    print(
        f"\nFINAL SUBMISSION:\n  {final_path}"
    )

    print(
        f"Five-seed target consistency discrepancy: "
        f"{ensemble_discrepancy:.12g}"
    )

    # -------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------
    disagreement = (
        seed_disagreement_table(
            seed_submissions
        )
    )

    disagreement.to_csv(
        METRICS_DIR
        / "test_seed_disagreement_by_row.csv",
        index=False,
    )

    disagreement_summary = (
        seed_disagreement_summary(
            disagreement
        )
    )

    disagreement_summary.to_csv(
        METRICS_DIR
        / "test_seed_disagreement_summary.csv",
        index=False,
    )

    distribution_summary = (
        prediction_distribution_summary(
            final_submission
        )
    )

    distribution_summary.to_csv(
        METRICS_DIR
        / "final_test_prediction_distribution.csv",
        index=False,
    )

    exp05_submission, comparison = (
        compare_to_exp05_submission(
            final_submission
        )
    )

    save_training_visualizations(
        histories
    )

    save_test_visualizations(
        final_submission,
        disagreement_summary,
        exp05_submission=(
            exp05_submission
        ),
    )

    save_top_peak_trajectories(
        final_submission
    )

    # -------------------------------------------------------------
    # Validation / final status
    # -------------------------------------------------------------
    validation = {
        "submission_file": str(
            final_path
        ),
        "model": (
            "TRIAL22_STATE_UOT_5SEED_FULL_DATA"
        ),
        "base_seeds": (
            BASE_SEEDS
        ),
        "actual_training_seeds": [
            int(
                s
                + FULL_TRAIN_SEED_OFFSET
            )
            for s in BASE_SEEDS
        ],
        "n_training_counties": int(
            len(
                train_ids
            )
        ),
        "n_test_counties": int(
            len(
                test_ids
            )
        ),
        "reach": UOT_REACH,
        "blur": UOT_BLUR,
        "diameter": UOT_DIAMETER,
        "lambda_uot": (
            FROZEN_LAMBDA_UOT
        ),
        "warmup_epochs": (
            WARMUP_EPOCHS
        ),
        "ramp_epochs": (
            RAMP_EPOCHS
        ),
        "epochs": EPOCHS,
        "batch_size": (
            BATCH_SIZE
        ),
        "learning_rate": (
            LR
        ),
        "five_seed_submission_target_consistency": (
            ensemble_discrepancy
        ),
        "phase11_blend_used": False,
        "final_configuration": "Frozen Optuna Trial 22 State-UOT",
        "trial22_selected_for_final_submission": True,
        "post_tuning_confirmation": (
            "five-seed x five-fold paired robustness audit completed; "
            "Trial 22 and incumbent essentially tied on competition-aligned RMSE"
        ),
        "previous_EXP05_test_submission_comparison_available": (
            comparison
            is not None
        ),
    }

    (
        OUT
        / "FINAL_SUBMISSION_STATUS.json"
    ).write_text(
        json.dumps(
            validation,
            indent=2,
        ),
        encoding="utf-8",
    )

    pd.DataFrame([
        {
            "submission": (
                FINAL_SUBMISSION_NAME
            ),
            "n_seeds": (
                len(
                    BASE_SEEDS
                )
            ),
            "base_seeds": (
                ",".join(
                    str(
                        x
                    )
                    for x in BASE_SEEDS
                )
            ),
            "actual_training_seeds": (
                ",".join(
                    str(
                        x
                        + FULL_TRAIN_SEED_OFFSET
                    )
                    for x in BASE_SEEDS
                )
            ),
            "max_duplicate_target_discrepancy": (
                ensemble_discrepancy
            ),
            **{
                f"nan_{col}": int(
                    final_submission[
                        col
                    ].isna().sum()
                )
                for col in base.TARGET_COLS
            },
        }
    ]).to_csv(
        METRICS_DIR
        / "FINAL_submission_validation.csv",
        index=False,
    )

    readme = f"""FINAL TRIAL-22 STATE-UOT SUBMISSION PACKAGE
==================================

FINAL SELECTED MODEL
--------------------
Frozen Trial-22 State-UOT five-seed full-data ensemble.

Final selection:
    Frozen Trial 22 State-UOT.
    No blending is used.

Trial-22 State-UOT:
    reach       = {UOT_REACH}
    blur        = {UOT_BLUR}
    diameter    = {UOT_DIAMETER}
    lambda_uot  = {FROZEN_LAMBDA_UOT:.15g}
    warm-up     = {WARMUP_EPOCHS}
    ramp        = {RAMP_EPOCHS}
    epochs      = {EPOCHS}
    batch size  = {BATCH_SIZE}
    learning rate = {LR}

Base seeds:
    {BASE_SEEDS}

Actual full-data seeds:
    {[x + FULL_TRAIN_SEED_OFFSET for x in BASE_SEEDS]}

Official final submission:
    submissions/{FINAL_SUBMISSION_NAME}

Physical-target consistency:
    {ensemble_discrepancy}

IMPORTANT
---------
Test labels are hidden. The final-vs-EXP05 comparison in this folder is only a
prediction-difference analysis, not an accuracy comparison.

Competition-ready files are under competition_ready/.
The written report must be added separately before submission.
Preserve this entire folder as the final reproducibility record.
"""

    (
        OUT
        / "00_README_FINAL_SUBMISSION.txt"
    ).write_text(
        readme,
        encoding="utf-8",
    )

    competition_prediction, code_zip_path, competition_discrepancy = (
        build_competition_ready_material(
            final_submission,
            final_path,
        )
    )

    print(
        f"\nCompetition prediction file:\n  {competition_prediction}"
    )
    print(
        f"Competition code archive:\n  {code_zip_path}"
    )
    print(
        f"Competition-copy target consistency discrepancy: "
        f"{competition_discrepancy:.12g}"
    )

    write_artifact_manifest()

    # -------------------------------------------------------------
    # Console summary
    # -------------------------------------------------------------
    print("\n" + "=" * 78)
    print(
        "PHASE 14 COMPLETE -- FINAL TRIAL-22 STATE-UOT SUBMISSION"
    )
    print("=" * 78)

    print(
        "\nSeed validation:"
    )
    print(
        seed_validation.to_string(
            index=False
        )
    )

    print(
        "\nAcross-seed test disagreement:"
    )
    print(
        disagreement_summary.to_string(
            index=False
        )
    )

    print(
        "\nFinal test prediction distribution:"
    )
    print(
        distribution_summary.to_string(
            index=False
        )
    )

    if comparison is not None:
        print(
            "\nState-UOT vs previous EXP05 final TEST predictions "
            "(NOT performance):"
        )

        print(
            comparison.to_string(
                index=False
            )
        )

    print(
        "\nFINAL DECISION:"
    )

    print(
        "  Submission model = FROZEN TRIAL-22 STATE-UOT five-seed full-data ensemble."
    )

    print(
        "  Phase-11 blend = NOT USED."
    )

    print(
        f"  Official file = {final_path}"
    )

    print(
        f"\nTotal elapsed: "
        f"{(time.time()-started)/60.0:.2f} minutes"
    )

    print(
        f"Saved package to:\n{OUT}"
    )


def main():
    log_path = (
        OUT
        / "PHASE14_TRIAL22_CONSOLE.txt"
    )

    original_stdout = (
        sys.stdout
    )

    original_stderr = (
        sys.stderr
    )

    with open(
        log_path,
        "w",
        encoding="utf-8",
        buffering=1,
    ) as log_file:
        sys.stdout = Tee(
            original_stdout,
            log_file,
        )

        sys.stderr = Tee(
            original_stderr,
            log_file,
        )

        try:
            run()

        except Exception:
            import traceback

            print(
                "\nPHASE 14 FAILED -- TRACEBACK"
            )

            traceback.print_exc()

            raise

        finally:
            sys.stdout = (
                original_stdout
            )

            sys.stderr = (
                original_stderr
            )


if __name__ == "__main__":
    main()
