# 🧠 Technical Knowledge Transfer: Talent Fraud Detection Agent
## Project: Eightfold AI "Agentic Among Us" Hackathon

This document summarizes the engineering decisions, iterations, and strategy shifts that led to the final **Peak-F1 (0.63)** Agent.

---

## 1. The Core Architecture
Our agent uses a **Three-Stage "Scout-Hunt-Ensemble"** framework.

### **Stage 1: Scout (Discovery Seeding)**
- **The Problem**: Random sampling only finds ~8 fraud cases in 100 queries. A model cannot learn a complex boundary with only 8 examples.
- **The Solution**: We used **Domain Expert Signals**. We 4-way targeted the specific fraud types mentioned in the README:
    1. **Credential Fraud**: `(institution_risk + gpa_anomaly + company_risk)`
    2. **App Bombing**: `(apps_7d / apps_30d + app_to_avg_ratio)`
    3. **Account Takeover**: `(is_new_device + failed_logins + login_velocity)`
    4. **Ghost Profiles**: `(copy_paste_ratio + profile_age)`
- **Why we did it**: By querying the "mathematical heart" of these signals first, we jumpstart the model with 30-40 confirmed fraud cases instead of 8.

### **Stage 2: Hunt (Iterative Active Learning)**
- **The Strategy**: We split the remaining budget into **5 rounds of 10 queries**.
- **The Innovation**: **Diversity-Weighted Acquisition**.
- **The Logic**: A standard model would keep querying the *same* cluster if it's "easy" to predict. We used **MiniBatchKMeans** on the top 150 fraud candidates in each round to pick the 10 most *geographically different* ones.
- **Why we did it**: This ensures the model expands the boundary for all 4 fraud types simultaneously rather than getting stuck in a single local "rabbit-hole."

### **Stage 3: Ensemble (Production Classification)**
- **The Model**: A weighted blend of **HistGradientBoosting (75%)** and **Random Forest (25%)**.
- **The Logic**: 
    - HistGBM is extremely good at finding complex tabular patterns (non-linear decision boundaries).
    - Random Forest is stable and prevents "high variance" errors on small datasets.
    - **Threshold 0.49**: We found that 0.50 (the default) was too conservative for Recall. Moving to 0.49 captured significantly more fraud without a major Precision penalty, causing the **F1 to peak**.

---

## 2. The Development Iterations (The Journey)

| Iteration | Logic | Result (F1) | Why we changed the idea |
| :--- | :--- | :--- | :--- |
| **v1: Random** | Random queries + Logistic Regression | ~0.15 | Too slow. Not enough fraud cases found to learn. |
| **v2: Pure Uncertainty** | Query points where p=0.5 | ~0.35 | The model was refining boundaries of areas it *thought* were fraud, but it was often wrong because it lacked "seeds". |
| **v3: Unsupervised Outliers** | Isolation Forest + KMeans Distance | ~0.31 | **CRITICAL FAILURE**: Every outlier is an outlier, but not every outlier is a FRAUD. It wasted queries on "Power Users" (weird but legitimate). |
| **v4: Surgical Seeding** | Hand-tuned signals for 4 Clusters | **~0.58** | **BREAKTHROUGH**: By targeting the *specific* fraud types, we shifted the labeled density from 8% to 45%. |
| **v5: Feature Augmentation**| Adding anomaly scores back into X | **~0.63** | By telling the model what *we* thought was suspicious, the GBDT became much faster at isolating the boundary. |

---

## 3. Key Concepts for Your Knowledge
1. **Budget-Constrained Optimization**: When labels are expensive (queries), **diversity** is more important than **confidence**. Don't ask the oracle for things you already know; ask for things that represent new territory.
2. **Cold Start Problem**: In Active Learning, the "Initial Seeding" determines your ceiling. A bad seed set leads to a model that can't see half the problem space.
3. **Threshold Tuning**: In F1 optimization, the default 0.50 is rarely binary-optimal. Always look at the Precision-Recall curve locally.
4. **Regularization (L2)**: With only 100 samples, models overfit instantly. We used high L2 regularization (L2=2.0+) to keep the decision boundaries "smooth."

---

## 4. Why this is a "Winner-Level" Agent
- It handles **Class Imbalance** via `compute_sample_weight`.
- It uses **Diversity Sampling** to maximize information gain per query.
- It is **Resilient**; it has fallbacks (if no fraud is found, it uses the unsupervised scores).
- It is **Surgical**; it knows exactly what "Eightfold AI Fraud" looks like.

*Prepared by CtrlAltReact AI Assistant (2026).*
