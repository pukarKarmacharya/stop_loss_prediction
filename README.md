# Dynamic Stop-Loss Prediction Using Deep Learning

This repository contains the implementation and trained artifacts for a deep learning system that predicts dynamic stop-loss levels from historical financial market data. The project focuses on improving the stability and interpretability of stop-loss prediction while remaining faithful to an established deep learning architecture from the literature.

A ready-to-use PyTorch model file (`.pt`) and the corresponding feature scaler (`.npz`) are provided for inference.

---

## Project Overview

Stop-loss mechanisms are widely used in trading systems to limit downside risk. Traditional approaches rely on fixed or rule-based thresholds, which do not adapt well to changing market conditions. Recent research has explored deep learning–based stop-loss prediction, but the formulation of the stop-loss learning objective remains challenging.

This project reproduces a published deep learning framework for stop-loss prediction and extends it through systematic reformulation of the stop-loss target. The goal is to improve numerical stability, decision clarity, and practical usability without changing the core architecture.

The system learns from historical price data and technical indicators using a supervised learning pipeline. It jointly models price behavior, market trends, and stop-loss signals.

---

## Major Technical Outcomes

The project delivers the following key outcomes:

- A faithful reproduction of a published deep learning–based stop-loss prediction model.
- Empirical analysis showing that direct stop-price regression leads to unstable or mean-biased behavior.
- Reformulation of stop-loss prediction using:
  - relative stop-loss representation,
  - classification-based stop decision learning,
  - bounded dynamic stop-loss magnitude prediction.
- A trained PyTorch model that produces stable and adaptive stop-loss outputs.
- A reusable inference pipeline that supports offline evaluation and future integration into trading systems.

These outcomes demonstrate that careful target formulation has a greater impact on stop-loss behavior than architectural complexity.

---

## Repository Structure (Key Files)

- `py_model/stoploss_pytorch_artifacts/`
  - `stoploss_model.pt`  
    Saved PyTorch model weights (`state_dict`).
  - `scaler.npz`  
    Saved feature scaler parameters (`cols`, `min_`, `max_`, `a`, `b`).

---

## Requirements

- Python 3.13 or later
- PyTorch
- NumPy
- pandas
- matplotlib (optional, for plotting)

Install dependencies using:

```bash
uv sync

---

## Reference Paper

This project is based on and extends the following published work:

**Samarasekara, I. K., Mendis, O. K., Ahangama, S., & Atukorale, A. S. (2022).**  
_Dynamic Stop-Loss Approach for Short Term Trades using Deep Learning._  
IEEE International Conference on Big Data.

The original paper introduces the baseline architecture and stop-loss target construction. This project reproduces that framework and proposes alternative formulations to improve stability, interpretability, and practical applicability.