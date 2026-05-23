# PBP2a QSAR Project

Modular scaffold-aware QSAR and docking pipeline for PBP2a inhibitor discovery.

# PBP2a QSAR Modeling and Cluster-Based Validation Framework

## Overview

This repository contains the complete computational workflow developed for the prediction and prioritization of potential PBP2a inhibitors using machine learning–based quantitative structure–activity relationship (QSAR) modeling.

The project was developed as part of a research study focused on identifying compounds with predicted anti-MRSA activity against Penicillin-Binding Protein 2a (PBP2a), a clinically important resistance determinant in methicillin-resistant *Staphylococcus aureus* (MRSA).

The workflow includes:-
- dataset curation,
- molecular standardization,
- Morgan fingerprint generation,
- Random Forest QSAR modeling,
- cluster-based external validation,
- applicability domain analysis,
- Y-randomization validation,
- residual diagnostics,
- publication-quality visualization generation.

---

# Key Features

- IC50-only standardized dataset
- Molecular standardization using RDKit
- 2048-bit Morgan fingerprints (ECFP-like)
- Random Forest regression model
- Cluster-based train/test splitting
- External validation on unseen chemical families
- Residual analysis
- Williams plot applicability domain analysis
- Y-randomization validation
- Automated figure and table generation
- Fully reproducible pipeline

---

# Repository Structure

```text
pbp2a_qsar_project/
│
├── Autodock/                  # External docking workflow/tool [Dockflow]
│
├── data/
│   └── raw/
│   └── splits/
│   └── processed/
│ 
├── models/
│
├── results/
│
├── scripts/
│
├── src/
│
├── README.md
│
└── requirements.txt
```

---

# Dataset Curation

The dataset was curated from publicly available bioactivity databases.

## Curation Workflow

- Removal of missing values
- Removal of invalid molecular structures
- Removal of duplicate SMILES
- Removal of censored activity records (`>`, `<`)
- Standardization of activity units
- Retention of IC50 endpoints only
- Conversion of IC50 values to pIC50

The pIC50 transformation used:

```math
pIC50 = -\log_{10}(IC50 \times 10^{-9})
```

---

# Molecular Representation

Compounds were encoded using:
- Morgan fingerprints
- Radius: 2
- Fingerprint size: 2048 bits

Generated using RDKit.

---

# Machine Learning Model

The final validated model used:
- RandomForestRegressor
- Cluster-based external validation
- External prediction on chemically distinct compound clusters

## Final Model Parameters

```python
RandomForestRegressor(
    n_estimators=1000,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)
```

---

# Validation Strategy

## Cluster-Based Validation

 compounds were separated using cluster-based validation to reduce chemical information leakage.

Workflow:
1. Generate Morgan fingerprints
2. Compute Tanimoto similarity
3. Cluster compounds into chemical families
4. Assign entire clusters to train or test sets

This produces a more realistic estimate of prospective model performance.

---

# Model Performance

Final external validation performance:

| Metric | Value |
|---|---|
| R² | ~0.52 |
| RMSE | ~0.74 |
| MAE | ~0.50 |

These values were obtained using cluster-based external validation.

---

# Validation Analyses Included

## Y-Randomization

- 100 permutations
- Statistical validation against chance correlation

## Residual Analysis

- Residual plots
- Prediction error diagnostics

## Applicability Domain

- Williams plot
- Leverage analysis
- Standardized residual analysis
- Identification of compounds outside applicability domain

---

# Generated Outputs

The pipeline automatically generates:

## Figures

- Predicted vs Actual pIC50
- pIC50 Distribution
- Residual Analysis
- Williams Plot
- Y-Randomization Plot

## Tables

- Predictions
- Residual values
- Applicability domain compounds
- Y-randomization statistics
- Validation metrics

---

# Docking Workflow and Outputs

The `Autodock/` directory contains the molecular docking inputs and outputs generated during structural validation and post-QSAR prioritization of predicted PBP2a inhibitors.

Docking was performed against the PBP2a crystal structure (PDB ID: 4CJN) to evaluate the predicted binding behavior of top-ranked QSAR candidates.

The docking workflow was performed using the separate [Dockflow](https://github.com/VajraPutra/Dockflow) framework.

## Directory Contents

### `Autodock Input/`
Contains:
- receptor structure files,
- input ligand datasets,
- docking preparation inputs.

Current contents include:
- `4CJN.pdb`
- `pbp2a_raw_data.csv`

### `Autodock Output/results/4CJN/`
Contains:
- docking poses,
- docking result files,
- AutoDock-generated scoring outputs,
- docking logs,
- prioritized hit tables.

Current contents include:
- `results.csv`
- `top_hits.csv`
- docking pose directories,
- receptor and ligand preparation outputs,
- docking logs.

The docking workflow involved:
- receptor preparation,
- ligand preparation,
- docking parameter configuration,
- pose generation,
- docking score analysis,
- interaction assessment.

Docking analysis was used as a complementary structural validation step and not as a standalone indicator of biological activity.

---

# Installation

## Clone Repository

```bash
git clone https://github.com/VajraPutra/pbp2a_qsar_project
cd pbp2a_qsar_project
```

---

## Create Environment

Linux/macOS:

```bash
python -m venv pbp2a_venv
source pbp2a_venv/bin/activate
```

Windows:

```bash
pbp2a_venv\Scripts\activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Pipeline

Place the curated dataset in:

```text
data/raw/pbp2a_raw_data.csv
```

Then run:

```bash
python scripts/run_pipeline.py
```

---

# Output Locations

## Figures

```text
results/figures/
```

## Tables

```text
results/tables/
```

## Trained Models

```text
models/trained/
```

## Metrics

```text
models/metrics/
```

---

# Requirements

Main dependencies:
- Python 3.10+
- RDKit
- scikit-learn
- pandas
- numpy
- matplotlib

---

# Reproducibility

This repository contains:
- raw datasets,
- processed datasets,
- scripts,
- trained models,
- validation workflows,
- plotting modules,
- reproducibility pipeline.

All computational analyses can be reproduced using the provided scripts.

---

# Scientific Disclaimer

This repository presents computational predictions only.

QSAR predictions, docking results, and physicochemical analyses do not constitute experimental validation or clinical evidence of efficacy. Experimental biochemical and microbiological validation is required before biological conclusions can be drawn.

---

# P.S

reference inhibitors docking results are added inside Autodock/reference_inhibitor_docking_result

---
# Citation

If you use this repository or workflow in academic work, please cite the associated research article.

---

# License

MIT License
