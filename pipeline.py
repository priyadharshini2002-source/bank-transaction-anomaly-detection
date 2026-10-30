"""Feature engineering + detectors shared by the Streamlit app (same logic as the notebook)."""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

SEED = 7
CITIES = {"Chennai": (13.08, 80.27), "Bengaluru": (12.97, 77.59), "Mumbai": (19.08, 72.88), "Delhi": (28.61, 77.21),
          "Hyderabad": (17.39, 78.49), "Kolkata": (22.57, 88.36), "Pune": (18.52, 73.86), "Coimbatore": (11.02, 76.96),
          "Thanjavur": (10.79, 79.14), "Madurai": (9.93, 78.12), "Singapore": (1.35, 103.82), "Dubai": (25.20, 55.27)}
CATS = ["grocery", "fuel", "restaurant", "online_retail", "pharmacy", "utilities", "travel", "electronics",
        "entertainment", "gambling_crypto"]
FEATURES = ["amount_robust_z", "log_amount", "hour_sin", "hour_cos", "hours_from_usual", "log_km_from_home",
            "category_rarity", "log_mins_since_prev", "log_speed", "tx_last_hour"]
DETECTORS = ["Ensemble", "Isolation Forest", "Local Outlier Factor", "One-Class SVM", "Robust z-score (amount)"]


def haversine(a, b):
    (la1, lo1), (la2, lo2) = np.radians(a).T, np.radians(b).T
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(h))


def load(path="data/simulated_transactions.csv.gz"):
    tx = pd.read_csv(path, parse_dates=["timestamp"])
    return tx.sort_values(["customer_id", "timestamp"]).reset_index(drop=True)


def profiles(tx, split):
    base = tx[tx.timestamp < split]
    prof = base.groupby("customer_id").agg(
        med_log=("amount_inr", lambda s: np.median(np.log1p(s))),
        mad_amt=("amount_inr", lambda s: stats.median_abs_deviation(np.log1p(s), scale="normal")),
        med_amt=("amount_inr", "median"),
        med_hour=("timestamp", lambda s: np.median(s.dt.hour + s.dt.minute / 60)),
        home_city=("city", lambda s: s.mode().iloc[0]),
        n_tx=("amount_inr", "size"))
    cat_share = (base.groupby(["customer_id", "merchant_category"]).size()
                 .div(base.groupby("customer_id").size(), level=0).rename("cat_share"))
    return prof, cat_share


def features(tx, prof, cat_share):
    f = tx.join(prof, on="customer_id")
    f["log_amount"] = np.log1p(f.amount_inr)
    f["amount_robust_z"] = (f.log_amount - f.med_log) / f.mad_amt.clip(lower=.1)
    hour = f.timestamp.dt.hour + f.timestamp.dt.minute / 60
    f["hour_sin"], f["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    f["hours_from_usual"] = np.minimum(abs(hour - f.med_hour), 24 - abs(hour - f.med_hour))
    coords = np.array([CITIES[c] for c in f.city]); homec = np.array([CITIES[c] for c in f.home_city])
    f["km_from_home"] = haversine(coords, homec)
    f["log_km_from_home"] = np.log1p(f.km_from_home)
    f = f.join(cat_share, on=["customer_id", "merchant_category"])
    f["category_rarity"] = 1 - f.cat_share.fillna(0)
    first = f.customer_id.ne(f.customer_id.shift()).values
    f["mins_since_prev"] = f.groupby("customer_id").timestamp.diff().dt.total_seconds().div(60).fillna(24 * 60).clip(upper=24 * 60)
    prev = np.roll(coords, 1, axis=0); prev[first] = coords[first]
    f["speed_kmh"] = (haversine(coords, prev) / (f.mins_since_prev.clip(lower=1) / 60)).clip(upper=5000)
    f["log_speed"] = np.log1p(f.speed_kmh)
    f["tx_last_hour"] = f.set_index("timestamp").groupby("customer_id").amount_inr.rolling("60min").count().values
    f["log_mins_since_prev"] = np.log1p(f.mins_since_prev)
    return f


def fit_and_score(f, split):
    rng = np.random.default_rng(SEED)
    train, test = f[f.timestamp < split], f[f.timestamp >= split].copy()
    scaler = StandardScaler().fit(train[FEATURES])
    Xtr, Xte = scaler.transform(train[FEATURES]), scaler.transform(test[FEATURES])
    iso = IsolationForest(n_estimators=300, random_state=SEED, n_jobs=-1).fit(Xtr)
    lof = LocalOutlierFactor(n_neighbors=50, novelty=True).fit(Xtr[rng.choice(len(Xtr), 15000, replace=False)])
    oc = OneClassSVM(gamma="scale", nu=0.01).fit(Xtr[rng.choice(len(Xtr), 6000, replace=False)])
    models = {"scaler": scaler, "Isolation Forest": iso, "Local Outlier Factor": lof, "One-Class SVM": oc}
    test["Isolation Forest"] = -iso.score_samples(Xte)
    test["Local Outlier Factor"] = -lof.score_samples(Xte)
    test["One-Class SVM"] = -oc.score_samples(Xte)
    test["Robust z-score (amount)"] = test.amount_robust_z.abs()
    test["Ensemble"] = np.mean([rankdata(test[m]) / len(test) for m in ["Isolation Forest", "Local Outlier Factor", "One-Class SVM"]], axis=0)
    # percentile of each score within the test period, used to rank new transactions
    return test, models


def score_new(row, history, prof, cat_share, models, ref):
    """Score one hypothetical transaction for a customer against the reference (test-period) scores."""
    tx = pd.concat([history, pd.DataFrame([row])], ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    f = features(tx, prof, cat_share)
    new = f[f.is_new.fillna(False).astype(bool)].iloc[[0]]
    X = models["scaler"].transform(new[FEATURES])
    out = {"Isolation Forest": -models["Isolation Forest"].score_samples(X)[0],
           "Local Outlier Factor": -models["Local Outlier Factor"].score_samples(X)[0],
           "One-Class SVM": -models["One-Class SVM"].score_samples(X)[0],
           "Robust z-score (amount)": abs(new.amount_robust_z.iloc[0])}
    pct = {k: float((ref[k].values < v).mean()) for k, v in out.items()}
    pct["Ensemble"] = float(np.mean([pct[m] for m in ["Isolation Forest", "Local Outlier Factor", "One-Class SVM"]]))
    z = pd.Series(X[0], index=FEATURES)
    return pct, new.iloc[0], z
