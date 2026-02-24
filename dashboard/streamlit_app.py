
import json
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="IIoT Mini Deploy", layout="wide")
st.title("🟢 IIoT Mini-Deploy — Ventana → Score / Anomalía")

MODELOS = {
    "0 Nm - ctvib SGD_Fused (fused_sgd_logistic_load0)": {
        "artifact": "fused_sgd_logistic_load0.joblib",
        "modality": "ctvib",
        "load_nm": 0,
        "desc": "F1=0.892, FP=49, recall=0.955"
    },
    "2 Nm - ct SGD_Logistic (ct_sgd_logistic_load2)": {
        "artifact": "ct_sgd_logistic_load2.joblib",
        "modality": "ct",
        "load_nm": 2,
        "desc": "F1=0.950, FP=2, recall=0.917"
    },
    "4 Nm - ct IsolationForest (ct_isolationforest_load4)": {
        "artifact": "ct_isolationforest_load4.joblib",
        "modality": "ct",
        "load_nm": 4,
        "desc": "F1=0.888, FP=5, recall=0.814"
    },
}

st.markdown("### Modelos disponibles para prueba")
df_info = pd.DataFrame([
    {"Carga": "0 Nm", "Modelo": "ctvib SGD_Fused", "F1": 0.892, "FP": 49, "Recall": 0.955},
    {"Carga": "2 Nm", "Modelo": "ct SGD_Logistic", "F1": 0.950, "FP": 2,  "Recall": 0.917},
    {"Carga": "4 Nm", "Modelo": "ct IsolationForest", "F1": 0.888, "FP": 5,  "Recall": 0.814},
])
st.dataframe(df_info, use_container_width=True)
st.divider()

with st.sidebar:
    api_url = st.text_input("API URL", value="http://127.0.0.1:8000")
    modelo_nombre = st.selectbox("Selecciona el modelo", list(MODELOS.keys()))
    modelo = MODELOS[modelo_nombre]
    st.caption(f"**{modelo['desc']}**")
    agg = st.selectbox("Agregación ventana", ["mean", "max"], index=0)

st.markdown("## Entrada: ventana de features (1Hz)")
st.caption("Asegúrate de que las columnas del JSON/CSV coincidan con las features del modelo seleccionado.")

colA, colB = st.columns([2, 1])
with colA:
    json_text = st.text_area("JSON samples", height=240, value="[]")

with colB:
    up = st.file_uploader("Sube CSV (columnas = features)", type=["csv"])
    if up is not None:
        df = pd.read_csv(up)
        st.write("Vista previa", df.head())
        json_text = df.to_json(orient="records")

if st.button("Enviar a /predict"):
    try:
        samples = json.loads(json_text)
        if not isinstance(samples, list):
            st.error("El JSON debe ser una lista.")
            st.stop()

        payload = {
            "modality": modelo["modality"],
            "load_nm": modelo["load_nm"],
            "samples": samples,
            "agg": agg,
            "artifact_file": modelo["artifact"],
        }

        r = requests.post(api_url.rstrip("/") + "/predict", json=payload, timeout=60)
        if r.status_code != 200:
            st.error(f"Error {r.status_code}: {r.text}")
            st.stop()

        out = r.json()
        st.success("OK")
        st.json(out)

        w = out.get("window", {})
        st.metric("Window pred", w.get("pred"))
        st.metric("Fault ratio", w.get("fault_ratio"))
        st.metric("Score mean", w.get("score_mean"))
        st.metric("Score max", w.get("score_max"))
        st.caption(f"Artifact: {out.get('artifact_file')} | thr={out.get('threshold')}")

    except Exception as e:
        st.error(str(e))
