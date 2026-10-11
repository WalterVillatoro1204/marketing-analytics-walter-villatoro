# Archivo base para el despliegue del Agente en Streamlit
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Laboratorio Uplift Starbucks", layout="wide")
st.title("Laboratorio: Modelo de Uplift y Políticas de Contacto")
st.write("Análisis de rentabilidad incremental usando T-Learner.")

# 1. Cargar datos
@st.cache_data
def cargar_datos():
    url = "https://raw.githubusercontent.com/WalterVillatoro1204/marketing-analytics-walter-villatoro/main/Semana_3/uplift_experiment_FULL.csv"
    df = pd.read_csv(url)
    return df

df = cargar_datos()
FEATURES = ["recency_days", "frequency", "monetary", "email_open_rate"]

# Separar train y test
train, test = train_test_split(df, test_size=0.30, random_state=42, stratify=df["treatment_group"])
test = test.copy()

# Entrenar T-learner
train_t = train[train["treatment_group"] == "treatment"]
train_c = train[train["treatment_group"] == "control"]

mt = LogisticRegression(max_iter=1000).fit(train_t[FEATURES], train_t["responded_90d"])
mc = LogisticRegression(max_iter=1000).fit(train_c[FEATURES], train_c["responded_90d"])

test["p_tratado"] = mt.predict_proba(test[FEATURES])[:, 1]
test["p_control"] = mc.predict_proba(test[FEATURES])[:, 1]
test["uplift_estimado"] = test["p_tratado"] - test["p_control"]

# 2. Control Interactivo (Slider)
st.sidebar.header("Configuración de Campaña")
pct_seleccionado = st.sidebar.slider("Porcentaje de la base a contactar (%):", min_value=10, max_value=100, value=40, step=10)

# Metricas segun el slider
n_contactar = int(len(test) * (pct_seleccionado / 100))
muestra = test.sort_values("uplift_estimado", ascending=False).head(n_contactar)
vt = muestra.loc[muestra["treatment_group"] == "treatment", "value_90d"].mean()
vc = muestra.loc[muestra["treatment_group"] == "control", "value_90d"].mean()
uplift_prom = vt - vc
total_ganancia = uplift_prom * n_contactar

col1, col2, col3 = st.columns(3)
col1.metric("Clientes Contactados", f"{n_contactar:,}")
col2.metric("Uplift Promedio por Cliente", f"${uplift_prom:.2f}")
col3.metric("Valor Incremental Total", f"${total_ganancia:,.2f}")

# 3. Graficar Curva Qini
st.subheader("1. Curva Qini del T-Learner")

def calcular_qini(datos, steps=40):
    orden = datos.sort_values("uplift_estimado", ascending=False).reset_index(drop=True)
    es_t = (orden["treatment_group"] == "treatment").values
    resp = orden["responded_90d"].values
    n = len(orden)
    xs, ys = [0.0], [0.0]
    for i in np.linspace(1, n, steps).astype(int):
        sub_t = resp[:i][es_t[:i]]
        sub_c = resp[:i][~es_t[:i]]
        tt = sub_t.mean() if len(sub_t) > 0 else 0
        tc = sub_c.mean() if len(sub_c) > 0 else 0
        xs.append(i / n)
        ys.append((tt - tc) * n)
    return np.array(xs), np.array(ys)

xs, ys = calcular_qini(test)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(xs, ys, label="T-Learner (Uplift)", color="#1f77b4", lw=2)
ax.plot([0, 1], [0, ys[-1]], linestyle="--", color="gray", label="Al Azar")
ax.set_xlabel("Fracción de la base contactada")
ax.set_ylabel("Respuestas incrementales acumuladas")
ax.legend()
st.pyplot(fig)

# 4. Tabla de politicas (Paso 3)
st.subheader("2. Tabla de Políticas por Deciles")

filas_tabla = []
for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    nc = int(len(test) * p)
    sub = test.sort_values("uplift_estimado", ascending=False).head(nc)
    v_t = sub.loc[sub["treatment_group"] == "treatment", "value_90d"].mean()
    v_c = sub.loc[sub["treatment_group"] == "control", "value_90d"].mean()
    u = v_t - v_c
    filas_tabla.append({
        "Porcentaje Base": f"{int(p*100)}%",
        "Clientes": nc,
        "Uplift por Cliente ($)": round(u, 2),
        "Valor Incremental Total ($)": round(u * nc, 2)
    })

st.dataframe(pd.DataFrame(filas_tabla), use_container_width=True)
