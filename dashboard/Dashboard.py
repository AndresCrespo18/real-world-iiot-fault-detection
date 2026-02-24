# dashboard_compare_v3.py
# Streamlit dashboard: compares Model 1 vs Model 2 vs Model 3 (CT+VIB fused)
# Compatible with artifacts produced by train_compare_final_dashboard_v13.py
#
# Run:
#   streamlit run dashboard_compare_v3.py
#
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import joblib
from pathlib import Path

st.set_page_config(page_title="IIoT — Model Comparison", layout="wide")
st.title("📊 IIoT — Comparación de Modelos (Model 1 vs Model 2 vs Model 3 CT+VIB)")

MODELS_DIR_DEFAULT = "models_out_final_v13"

def cm_fig(cm, title):
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=["Pred Normal", "Pred Fault"],
        y=["True Normal", "True Fault"],
        colorscale="Blues"
    ))
    fig.update_layout(title=title, height=320, margin=dict(l=10, r=10, t=40, b=10))
    return fig

def metrics_table(m):
    keep = ["accuracy","precision","recall","f1","roc_auc","avg_precision","tn","fp","fn","tp"]
    rows = [(k, m.get(k, "N/A")) for k in keep]
    return pd.DataFrame(rows, columns=["metric","value"])

def top_reasons_from_z(x, baseline, topk=3):
    if not baseline:
        return []
    cols = baseline.get("feature_cols", [])
    med = np.array(baseline.get("median", []), dtype=float)
    scale = np.array(baseline.get("scale", []), dtype=float)
    if len(cols) != len(med) or len(cols) != len(scale) or len(cols) == 0:
        return []
    z = np.abs((x - med) / (scale + 1e-12))
    idx = np.argsort(-z)[:topk]
    return [(cols[i], float(z[i])) for i in idx]

def model1_predict(pipe, thr, x_row, score_sign=1):
    X = np.asarray(x_row, dtype=float).reshape(1, -1)
    X_imp = pipe.named_steps["imputer"].transform(X)
    X_sc = pipe.named_steps["scaler"].transform(X_imp)
    score = float(-pipe.named_steps["iforest"].score_samples(X_sc)[0])
    score = float(score_sign) * score
    pred = int(score >= thr)
    return pred, score

def model23_predict(clf, scaler, thr, x_row, score_sign=1, imputer=None):
    X = np.asarray(x_row, dtype=float).reshape(1, -1)
    if imputer is not None:
        X = imputer.transform(X)
    X_sc = scaler.transform(X)
    p = float(clf.predict_proba(X_sc)[0, 1])
    if int(score_sign) == -1:
        p = 1.0 - p
    pred = int(p >= thr)
    return pred, p

def epoch_fig(history, title):
    if not history:
        return None
    fig = None
    if isinstance(history, dict) and "val_f1" in history:
        epochs = list(range(1, len(history["val_f1"]) + 1))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=epochs, y=history["val_f1"], mode="lines+markers", name="val_f1"))
        if "val_auc" in history:
            fig.add_trace(go.Scatter(x=epochs, y=history["val_auc"], mode="lines+markers", name="val_auc"))
        if "val_ap" in history:
            fig.add_trace(go.Scatter(x=epochs, y=history["val_ap"], mode="lines+markers", name="val_ap"))
    elif isinstance(history, list) and len(history) > 0 and isinstance(history[0], dict):
        hdf = pd.DataFrame(history)
        epochs = hdf["epoch"] if "epoch" in hdf.columns else list(range(1, len(hdf)+1))
        fig = go.Figure()
        for col in ["val_f1","val_auc","val_ap","val_acc"]:
            if col in hdf.columns:
                fig.add_trace(go.Scatter(x=epochs, y=hdf[col], mode="lines+markers", name=col))
    if fig is not None:
        fig.update_layout(title=title, height=320, margin=dict(l=10, r=10, t=40, b=10))
    return fig

with st.sidebar:
    models_dir = st.text_input("Carpeta de modelos", value=MODELS_DIR_DEFAULT)
    p = Path(models_dir)
    if not p.exists():
        st.warning("La carpeta no existe. Escribe la ruta correcta.")
        st.stop()

    files = sorted(p.glob("*.joblib"))
    if not files:
        st.warning("No se encontraron .joblib en la carpeta.")
        st.stop()

    sel_file = st.selectbox("Selecciona un artifact (.joblib)", files, format_func=lambda x: x.name)

artifact = joblib.load(sel_file)

split_info = artifact.get("split_groups", {})

meta = artifact.get("meta", {})
mod = meta.get("modality", "?")
load_nm = meta.get("load_nm", "?")
feat_cols = meta.get("feature_cols", [])
asset = meta.get("asset", "?")

m1 = artifact.get("models", {}).get("model1")
m2 = artifact.get("models", {}).get("model2")
m3 = artifact.get("models", {}).get("model3")

m1_meta = artifact.get("model1", {})
m2_meta = artifact.get("model2", {})
m3_meta = artifact.get("model3", {})

metrics = artifact.get("metrics", {})
m1_metrics = metrics.get("model1", {})
m2_metrics = metrics.get("model2", {})
m3_metrics = metrics.get("model3", {})

history2 = artifact.get("history", {}).get("sgd", {})
history3 = artifact.get("history", {}).get("sgd3", {})

pre2 = artifact.get("preprocess", {})
pre3 = artifact.get("preprocess3", {})

st.subheader(f"Modality: **{str(mod).upper()}** | Load: **{load_nm}Nm** | Asset: **{asset}**")

if split_info:
    st.caption(f"Split: {split_info.get('notes','')} | mode={split_info.get('mode','')}")

cols_n = 3 if m3 is not None and pre3 else 2
cols = st.columns(cols_n)

# -------- Model 1 --------
with cols[0]:
    st.markdown("### Model 1 — IsolationForest")
    if m1 is None:
        st.info("No disponible.")
    else:
        st.dataframe(metrics_table(m1_metrics), use_container_width=True)
        cm = np.array([[m1_metrics.get("tn",0), m1_metrics.get("fp",0)],
                       [m1_metrics.get("fn",0), m1_metrics.get("tp",0)]])
        st.plotly_chart(cm_fig(cm, "Confusion Matrix — Model 1"), use_container_width=True)

# -------- Model 2 --------
with cols[1]:
    st.markdown("### Model 2 — SGD Logistic")
    if m2 is None or not pre2:
        st.info("No disponible.")
    else:
        st.dataframe(metrics_table(m2_metrics), use_container_width=True)
        cm = np.array([[m2_metrics.get("tn",0), m2_metrics.get("fp",0)],
                       [m2_metrics.get("fn",0), m2_metrics.get("tp",0)]])
        st.plotly_chart(cm_fig(cm, "Confusion Matrix — Model 2"), use_container_width=True)
        fig = epoch_fig(history2, "Epoch curve (validation) — Model 2")
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)

# -------- Model 3 --------
if cols_n == 3:
    with cols[2]:
        st.markdown("### Model 3 — CT+VIB (fused) — SGD Logistic")
        st.dataframe(metrics_table(m3_metrics), use_container_width=True)
        cm = np.array([[m3_metrics.get("tn",0), m3_metrics.get("fp",0)],
                       [m3_metrics.get("fn",0), m3_metrics.get("tp",0)]])
        st.plotly_chart(cm_fig(cm, "Confusion Matrix — Model 3"), use_container_width=True)
        fig = epoch_fig(history3, "Epoch curve (validation) — Model 3")
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)

st.divider()
st.markdown("## Prueba manual: introduce valores y compara respuestas")

# Manual inputs for Model 1/2 (current modality)
defaults = {c: 0.0 for c in feat_cols}
cols_in = st.columns(3)
vals = []
for i, f in enumerate(feat_cols):
    col = cols_in[i % 3]
    vals.append(col.number_input(f, value=float(defaults[f]), format="%.6f", key=f"m12_{f}"))

if st.button("Evaluar punto manual (Model 1/2)"):
    x = np.array(vals, dtype=float)

    cA, cB = st.columns(2)

    with cA:
        st.markdown("### Resultado Model 1")
        if m1 is None:
            st.info("No disponible.")
        else:
            thr = float(m1_meta.get("threshold", 0.5))
            sign = int(m1_meta.get("score_sign", 1))
            pred, score = model1_predict(m1, thr, x, score_sign=sign)
            st.metric("Predicción", "FAULT" if pred==1 else "NORMAL")
            st.metric("Anomaly score", f"{score:.5f}")
            st.metric("Threshold", f"{thr:.5f}")
            baseline = m1_meta.get("baseline")
            reasons = top_reasons_from_z(x, baseline, topk=3)
            if reasons:
                st.write("Top razones (robust z):", reasons)

    with cB:
        st.markdown("### Resultado Model 2")
        if m2 is None or not pre2:
            st.info("No disponible.")
        else:
            thr = float(m2_meta.get("threshold", 0.5))
            sign = int(m2_meta.get("score_sign", 1))
            scaler = pre2.get("scaler")
            imputer = pre2.get("imputer")
            pred, prob = model23_predict(m2, scaler, thr, x, score_sign=sign, imputer=imputer)
            st.metric("Predicción", "FAULT" if pred==1 else "NORMAL")
            st.metric("Prob fault (orientada)", f"{prob:.5f}")
            st.metric("Threshold", f"{thr:.5f}")
            baseline = m2_meta.get("baseline")
            reasons = top_reasons_from_z(x, baseline, topk=3)
            if reasons:
                st.write("Top razones (robust z):", reasons)

if cols_n == 3:
    with st.expander("Modelo 3 (CT+VIB) — Prueba manual (inputs fused)", expanded=False):
        feat3 = pre3.get("feature_cols", [])
        if not feat3:
            st.info("No hay feature_cols para Model 3.")
        else:
            defaults3 = {c: 0.0 for c in feat3}
            cols3 = st.columns(3)
            vals3 = []
            for i, f in enumerate(feat3):
                col = cols3[i % 3]
                vals3.append(col.number_input(f, value=float(defaults3[f]), format="%.6f", key=f"m3_{f}"))
            if st.button("Evaluar punto manual (Model 3)"):
                x3 = np.array(vals3, dtype=float)
                thr = float(m3_meta.get("threshold", 0.5))
                sign = int(m3_meta.get("score_sign", 1))
                scaler = pre3.get("scaler")
                imputer = pre3.get("imputer")
                pred, prob = model23_predict(m3, scaler, thr, x3, score_sign=sign, imputer=imputer)
                st.metric("Predicción", "FAULT" if pred==1 else "NORMAL")
                st.metric("Prob fault (orientada)", f"{prob:.5f}")
                st.metric("Threshold", f"{thr:.5f}")
                baseline = m3_meta.get("baseline")
                reasons = top_reasons_from_z(x3, baseline, topk=3)
                if reasons:
                    st.write("Top razones (robust z):", reasons)
