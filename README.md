# het-GPR4Materials

**Heteroscedastic Gaussian process Regression (GPR) for Atomistic Materials Modeling with Uncertainty Quantification**

This repository provides a Gaussian process regression framework for predicting density functional theory (DFT) quantities—such as energies and forces—with:

- Predictive **uncertainty quantification**
- Support for **input-dependent noise**
- Accurate energy and force predictions using **SOAP descriptors**
- Modular design using PyTorch and Metatensor

---

## Motivation

Traditional atomistic machine learning models assume:
- **Homoscedastic noise**: constant uncertainty across configurations
- **No uncertainty estimates**: models give predictions but not confidence

This project addresses these gaps by introducing:
- **Per-structure and per-atom noise levels** (heteroscedasticity)
- **Predictive variances** using the GPR kernel posterior
- An error-informed approach that enables more **efficient** and flexible use of **multi-fidelity** data
- Built-in **uncertainty quantification**, which supports more efficient active learning workflows

---

## Features

- Gaussian process regression using the **Subset of Regressors (SoR)** method
- Efficient SOAP vectorization with gradients using `featomic`
- Energy and force fitting with automatic handling of gradients
- Modular training and evaluation with YAML-configurable hyperparameters

---

## Installation

```bash
git clone https://github.com/yourusername/het-GPR4Materials.git
cd het-GPR4Materials
pip install -r requirements.txt
```
## File structure

```bash
.
├── gpr.py               # Core GPR model and kernel solver
├── gpr_example.ipynb    # End-to-end training + evaluation notebook
├── options.yaml         # Model, training, and dataset configuration
├── data/                # Train and test data
├── report/              # Report and slides
└── README.md
```
