"""Streamlit app — Bank Transaction Anomaly Detection.

Run locally:  pip install -r requirements.txt && streamlit run app.py
"""
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import roc_auc_score, average_precision_score

import pipeline as pl

st.set_page_config(page_title="Bank Transaction Anomaly Detection", page_icon="🏦", layout="wide")


@st.cache_resource(show_spinner="Building features and fitting detectors (about 30 s on first load)…")
def build():
    tx = pl.load()
    split = tx.timestamp.min().normalize() + pd.Timedelta(days=30)
    prof, cat_share = pl.profiles(tx, split)
    f = pl.features(tx, prof, cat_share)
    test, models = pl.fit_and_score(f, split)
    return tx, split, prof, cat_share, test, models


tx, split, prof, cat_share, test, models = build()
y = test.is_fraud.values

st.title("🏦 Bank Transaction Anomaly Detection")
st.caption("Unsupervised detectors fitted on days 1–30 of a simulated 143K-transaction stream, scoring days 31–60. "
           "Fraud labels are used only to measure performance.")

tab1, tab2, tab3 = st.tabs(["🚨 Alert queue", "🧪 Score a new transaction", "📊 Detector comparison"])

with tab1:
    c1, c2 = st.columns([1, 2])
    det = c1.selectbox("Detector", pl.DETECTORS)
    budget = c2.slider("Alert budget: % of transactions the team can review", 0.1, 3.0, 1.0, 0.1)
    k = int(round(budget / 100 * len(test)))
    q = test.sort_values(det, ascending=False).head(k)
    tp = int(q.is_fraud.sum())
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Alerts", f"{k:,}")
    m2.metric("Genuine fraud in alerts", f"{tp:,}")
    m3.metric("Precision", f"{tp / k:.0%}")
    m4.metric("Fraud caught (recall)", f"{tp / y.sum():.0%}")
    show = q[["customer_id", "timestamp", "amount_inr", "merchant_category", "city", "home_city",
              "km_from_home", "tx_last_hour", det, "fraud_type"]].copy()
    show["timestamp"] = show.timestamp.dt.strftime("%d %b %H:%M")
    show = show.rename(columns={det: "anomaly score", "fraud_type": "actual (hidden from model)"})
    st.dataframe(show.head(300).round(2), use_container_width=True, hide_index=True)
    st.caption("Showing up to 300 highest-scoring alerts.")

with tab2:
    st.write("Create a transaction for an existing customer and see how unusual it looks **for that customer**.")
    a, b, c = st.columns(3)
    cid = a.number_input("Customer ID", 0, int(tx.customer_id.max()), 42)
    p = prof.loc[cid]
    a.info(f"Home: **{p.home_city}** · usual amount ≈ **₹{p.med_amt:,.0f}** · usual time ≈ **{int(p.med_hour):02d}:00**")
    amount = b.number_input("Amount (₹)", 1.0, 1_000_000.0, float(round(p.med_amt * 6)), step=100.0)
    cat = b.selectbox("Merchant category", pl.CATS, index=pl.CATS.index("electronics"))
    city = c.selectbox("City", list(pl.CITIES), index=list(pl.CITIES).index("Dubai"))
    hour = c.slider("Hour of day", 0, 23, 2)
    hist = tx[(tx.customer_id == cid) & (tx.timestamp < split)].copy()
    when = hist.timestamp.max() + pd.Timedelta(days=1)
    when = when.normalize() + pd.Timedelta(hours=hour)
    row = {"customer_id": cid, "timestamp": when, "amount_inr": amount, "merchant_category": cat, "city": city,
           "channel": "online", "fraud_type": "?", "is_fraud": 0, "is_new": True}
    pct, feat, z = pl.score_new(row, hist.assign(is_new=False), prof, cat_share, models, test)
    e = pct["Ensemble"]
    level = "🔴 HIGH RISK — send to review" if e >= .99 else ("🟠 Elevated" if e >= .95 else "🟢 Looks normal")
    st.subheader(f"{level}  ·  ensemble percentile {e:.1%}")
    st.progress(min(e, 1.0))
    cols = st.columns(4)
    for col, name in zip(cols, ["Isolation Forest", "Local Outlier Factor", "One-Class SVM", "Robust z-score (amount)"]):
        col.metric(name, f"{pct[name]:.1%}", help="Share of test-period transactions that scored lower")
    st.write("**Why?** Features furthest from normal behaviour (standardised):")
    st.bar_chart(z.reindex(z.abs().sort_values(ascending=False).index).head(6))

with tab3:
    rows = []
    for d in pl.DETECTORS:
        s = test[d].values
        kk = int(round(.01 * len(s))); top = np.argsort(-s)[:kk]
        rows.append({"detector": d, "ROC-AUC": roc_auc_score(y, s), "PR-AUC": average_precision_score(y, s),
                     "precision @1%": y[top].mean(), "recall @1%": y[top].sum() / y.sum()})
    st.dataframe(pd.DataFrame(rows).sort_values("PR-AUC", ascending=False).round(3), use_container_width=True, hide_index=True)
    st.write("**Recall by fraud type at a 1% alert budget**")
    rec = {}
    for d in pl.DETECTORS:
        kk = int(round(.01 * len(test))); flagged = test[d].rank(ascending=False) <= kk
        rec[d] = test[test.is_fraud == 1].assign(f=flagged).groupby("fraud_type").f.mean()
    st.dataframe(pd.DataFrame(rec).T.style.format("{:.0%}").background_gradient(cmap="RdYlGn", vmin=0, vmax=1),
                 use_container_width=True)
    st.caption("The app uses a lighter version of the notebook's detectors so it loads quickly. "
               "See the notebook for the full analysis, including Mahalanobis + χ² and the bootstrap significance test.")
