# 🚀 CtrlAltReact — Talent Fraud Detection Agent
### **🏆 Eightfold AI "Agentic Among Us" Hackathon**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Table of Contents
- [Overview](#overview)
- [The Challenge](#the-challenge)
- [Our Winning Strategy: Peak-F1](#our-winning-strategy-peak-f1)
- [Performance Results](#performance-results)
- [Technical Architecture](#technical-architecture)
- [How to Run](#how-to-run)
- [Team](#team)

---

## 🔍 Overview
**CtrlAltReact** is a high-performance Active Learning agent designed for the **Eightfold AI Talent Fraud Detection Challenge**. In an environment where fraudulent job applications (fake credentials, bot bombing, hijacked accounts) are rare (~8%), our agent identifies fraud with surgical precision using a limited budget of only **100 labels**.

## ⚔️ The Challenge
- **Dataset**: 10,000 unlabeled candidate profiles.
- **Budget**: Only 100 oracle queries (1% of the total data).
- **Goal**: Maximize **F1 Score** across heterogeneous fraud clusters.
- **Constraints**: Pure Active Learning environment—no pre-labeled data allowed.

---

## 🧠 Our Winning Strategy: Peak-F1
We implemented a multi-stage **Discover-Refine-Ensemble** architecture that maximizes every single oracle query.

### **Phase 1: Multi-Modal Discovery (50% Budget)**
We found that uniform random sampling is inefficient. Instead, we use expert-system signals to "seed" the model:
- **Domain Peaks (32 Queries)**: Targeted the mathematical "hearts" of the four known fraud clusters (Credential Fraud, App Bombing, Account Takeover, Ghost Profiles).
- **Global Outliers (6 Queries)**: Used an `IsolationForest` to catch anomalous behaviors that don't fit standard patterns.
- **Structural Grounding (12 Queries)**: Used `K-Means` centroids to map the legitimate baseline, protecting our **Precision**.

### **Phase 2: Diversified Active Exploitation (50% Budget)**
We retrain the model in **5 iterative rounds**. In each round:
1. The model predicts fraud probabilities for all 10,000 rows.
2. We cluster the top 150 candidates.
3. We select the most **geographically diverse** candidates from different clusters.
4. This prevent the agent from "rabbit-holing" into a single fraud group and ensures we cover the entire fraud manifold.

### **Phase 3: Final Production Ensemble**
Our final prediction uses a weighted blend:
- **75% HistGradientBoosting**: Optimized for complex tabular irregularities.
- **25% Random Forest**: Provides local stability and prevents over-fitting on small samples.
- **Decision Threshold (0.49)**: Specifically calibrated to maximize the F1 metric by balancing Precision and Recall.

---

## 📊 Performance Results
Tested locally using the provided validation framework:

| Metric | Score |
| :--- | :--- |
| **F1 Score** | **0.6301** |
| **Precision** | **0.6432** |
| **Recall** | **0.6175** |
| **Budget Efficiency** | **99% (99/100 Queries)** |

---

## 🛠️ Technical Architecture
- **Classifier**: `HistGradientBoostingClassifier`, `RandomForestClassifier`.
- **Active Learning**: Uncertainty sampling with K-Means diversity filtering.
- **Preprocessing**: `StandardScaler` + Custom Signal Engineering.
- **Optimization**: L2 Regularization to handle high variance in low-label regimes.

---

## 🚀 How to Run
1. **Clone the Repo**:
   ```bash
   git clone https://github.com/your-username/eightfold-hackathon.git
   cd eightfold-hackathon
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run Evaluation**:
   ```bash
   python framework.py --agent agent.py
   ```

---

## 👥 Team: CtrlAltReact
- **Poojasri** — Lead (poojasrinirmalamanickam@gmail.com)
- **Deepthi** — Member (deepthitheeran@gmail.com)

*Developed for the Eightfold AI Agentic Among Us Hackathon (2026).*
