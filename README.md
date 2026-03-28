
# XTR: Explainable Transformer-Based Retracker for CryoSat-2 Altimetry

**Authors:**  
- Alireza Dehghanpour  
- Danial Barazandeh  
- Gabriel Zachmann  

> Developed at [University of Bremen – Computer Graphics and Virtual Reality Research Lab (CGVR Lab)](https://cgvr.cs.uni-bremen.de/)

---

![XTR Architecture](./figures/xtr_architecture.png)

---

## 📚 Overview

XTR is an explainable Transformer-based retracker for CryoSat-2 Low Resolution Mode (LRM) radar waveforms over polar ice sheets.

It combines:
- a **CNN stem** for local feature extraction (leading edge)
- a **Transformer encoder** for global waveform context
- **Explainable AI (XAI)** tools for interpretability

The goal is to reduce radar penetration bias and improve elevation consistency while maintaining physical interpretability.

---

## ✨ Highlights

- First **Transformer-based retracker** for CryoSat-2 LRM waveforms  
- Combines **local (CNN)** + **global (attention)** modeling  
- Provides **physically interpretable predictions**  
- Includes built-in **Explainable AI (XAI)** tools  

---

## 🧠 Key Idea

Unlike CNN-based retrackers (e.g., AWI-ICENet1), XTR introduces:

- 🔍 **Self-attention** → models long-range dependencies
- 📊 **Interpretability** → attention + saliency
- 🧊 **Physics-consistent behavior** → focuses on leading edge

---

## 📊 Example Explainability

![XAI Example](./figures/xai_example.png)

- (a) Input waveform + reference
- (b) SmoothGrad saliency
- (c) Surface-query attention

The model clearly focuses on the **leading edge**, while down-weighting volume scattering.

---

## 🚀 Key Features

- 🧠 Hybrid CNN + Transformer architecture
- 📈 Trained on simulated CryoSat-2 LRM waveforms
- 🔄 Physically motivated augmentation:
  - noise
  - amplitude jitter
  - temporal shifts
  - smoothing
- 📊 Metrics:
  - MSE
  - RMSE
  - MAE
  - R²
- 🔍 Explainability:
  - SmoothGrad
  - Attention maps
  - Surface-query attention

---

## ⚠️ Notice

The full code will be released after publication.

📧 Contact: a.r.dehghanpour@gmail.com

---

## 💻 Installation

```bash
git clone https://github.com/Alireza-Dehghanpour/xtr-retracker.git
cd xtr-retracker

pip install -r requirements.txt
```

## ⚙️ Usage

```bash
# Train the model
python train.py
```

---

## 📁 Folder Structure

```
xtr-retracker/
├── figures/
│   ├── xtr_architecture.png
│   ├── xai_example.png
├── src/
│   ├── data_loader.py
│   ├── model.py
│   ├── augmentations.py
│   ├── metrics.py
│   ├── xai.py
│   ├── config.py
├── train.py
├── README.md
├── requirements.txt
├── environment.yml
└── .gitignore
```
---


## 🌐 Data

You can download the full simulated reference dataset, elevation change grids, and monthly elevation anomalies from the World Data Center PANGAEA:

[https://doi.org/10.1594/PANGAEA.964596](https://doi.org/10.1594/PANGAEA.964596)

---

## 👨‍💻 Developer Notes

🧩 **Code developed by Alireza Dehghanpour, 2025.**  

📧 Contact: [a.r.dehghanpour@gmail.com](mailto:a.r.dehghanpour@gmail.com)

---
