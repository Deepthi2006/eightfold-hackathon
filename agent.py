import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.utils import compute_sample_weight
from sklearn.cluster import MiniBatchKMeans

def _get_anomaly_scores(df: pd.DataFrame) -> np.ndarray:
    """Core anomaly dimensions for Eightfold AI fraud clusters (F1 Optimized)."""
    # Signals derived from Eightfold AI README - Surgical Targeting
    c1 = (df["institution_risk_score"] + df["gpa_anomaly_score"] + df["company_risk_score"] + df["tenure_gap_months"]/20.0).rank(pct=True)
    c2 = (df["applications_7d"]/10.0 + df["applications_30d"]/30.0 + df["app_to_avg_ratio"] - df["time_since_last_app_hrs"]).rank(pct=True)
    c3 = (df["email_risk_score"] + df["is_new_device"] + df["failed_logins_24h"] + df["login_velocity_24h"]/5.0).rank(pct=True)
    c4 = (df["copy_paste_ratio"] + df["skills_to_exp_ratio"]/10.0 - df["profile_age_days"] / 500.0).rank(pct=True)
    return np.stack([c1.values, c2.values, c3.values, c4.values], axis=1)

def run_agent(df: pd.DataFrame, oracle_fn, budget: int) -> np.ndarray:
    n = len(df)
    X_raw = df.drop(columns=["feature_noise_1", "feature_noise_2", "feature_noise_3", "feature_noise_4"], errors='ignore')
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_raw.select_dtypes(include=[np.number]).fillna(0))
    
    # Feature Augmentation
    scores = _get_anomaly_scores(df)
    X_aug = np.hstack([X_sc, scores])
    
    labeled_idx = []
    labeled_y = []
    
    def query(indices):
        nonlocal budget
        to_q = [int(i) for i in indices if i not in set(labeled_idx)]
        if not to_q: return
        rem = budget - len(labeled_idx)
        if rem <= 0: return
        query_batch = to_q[:rem]
        res = oracle_fn(query_batch)
        labeled_idx.extend(query_batch)
        labeled_y.extend(res)

    # --- PHASE 1: DISCOVERY (50 queries) ---
    # Top 8 per domain cluster = 32
    for d in range(4):
        query(np.argsort(-scores[:, d])[:8].tolist())
    # 6 Outliers from Isolation Forest
    iso = IsolationForest(contamination=0.1, random_state=42).fit(X_sc)
    query(np.argsort(iso.decision_function(X_sc))[:6].tolist())
    # 12 K-Means structural anchors
    km_pop = MiniBatchKMeans(n_clusters=12, n_init=10, random_state=42).fit(X_sc)
    query([np.where(km_pop.labels_ == i)[0][0] for i in range(12)])

    # --- PHASE 2: DIVERSIFIED ACTIVE EXPLOITATION (50 queries) ---
    for _ in range(5):
        rem = budget - len(labeled_idx)
        if rem <= 0: break
        
        X_tr, y_tr = X_aug[labeled_idx], np.array(labeled_y)
        if len(np.unique(y_tr)) > 1:
            sw = compute_sample_weight("balanced", y_tr)
            m = HistGradientBoostingClassifier(max_iter=100, max_depth=3, random_state=42).fit(X_tr, y_tr, sample_weight=sw)
            proba = m.predict_proba(X_aug)[:, 1]
        else:
            proba = scores.max(axis=1)
            
        unlabeled = np.array([i for i in range(n) if i not in set(labeled_idx)])
        # Diversity-Weighted Hunting
        top_cand_idx = unlabeled[np.argsort(-proba[unlabeled])[:150]]
        km_q = MiniBatchKMeans(n_clusters=min(len(top_cand_idx), 10), n_init=10, random_state=42).fit(X_sc[top_cand_idx])
        batch_to_q = []
        for c in range(km_q.n_clusters):
            c_pts = np.where(km_q.labels_ == c)[0]
            if len(c_pts) > 0:
                batch_to_q.append(int(top_cand_idx[c_pts[np.argmax(proba[top_cand_idx[c_pts]])]]))
        query(batch_to_q)

    # --- PHASE 3: FINAL ENSEMBLE ---
    X_f, y_f = X_aug[labeled_idx], np.array(labeled_y)
    
    if len(np.unique(y_f)) > 1:
        sw_f = compute_sample_weight("balanced", y_f)
        m1 = HistGradientBoostingClassifier(max_iter=500, max_depth=5, learning_rate=0.03, l2_regularization=2.0, random_state=42).fit(X_f, y_f, sample_weight=sw_f)
        m2 = RandomForestClassifier(n_estimators=500, max_depth=16, class_weight="balanced", random_state=42).fit(X_f, y_f)
        
        p_final = m1.predict_proba(X_aug)[:, 1] * 0.75 + m2.predict_proba(X_aug)[:, 1] * 0.25
        return (p_final >= 0.49).astype(int)
    else:
        # Emergency Fallback
        preds = np.zeros(n, dtype=int)
        preds[np.argsort(-scores.max(axis=1))[:int(0.08 * n)]] = 1
        return preds