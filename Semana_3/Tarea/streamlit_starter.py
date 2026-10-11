"""
Laboratorio Final — Starbucks Rewards
App de Streamlit interactiva para explorar Qini y tabla de políticas de contacto.

Correr con:  streamlit run streamlit_starter.py
"""

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Laboratorio Final — Starbucks Rewards", page_icon="☕", layout="wide")

GREEN = "#00704A"
ORANGE = "#C4512F"
GRAY = "#9A988E"
FEATURES = ["recency_days", "frequency", "monetary", "email_open_rate", "tenure_days"]


@st.cache_data
def load_data():
    try:
        return pd.read_csv("data/uplift_campaign.csv")
    except Exception:
        url = "https://raw.githubusercontent.com/WalterVillatoro1204/marketing-analytics-walter-villatoro/main/Semana_3/Tarea/data/uplift_campaign.csv"
        return pd.read_csv(url)


@st.cache_data
def entrenar_t_learner(df: pd.DataFrame, test_size: float, seed: int):
    train, test = train_test_split(df, test_size=test_size, random_state=seed, stratify=df["treatment_group"])

    # 1. Modelos del T-learner
    train_t = train[train["treatment_group"] == "treatment"]
    train_c = train[train["treatment_group"] == "control"]
    
    modelo_tratado = LogisticRegression(max_iter=1000).fit(train_t[FEATURES], train_t["responded_60d"])   
    modelo_control = LogisticRegression(max_iter=1000).fit(train_c[FEATURES], train_c["responded_60d"])

    test = test.copy()
    X_test = test[FEATURES]
    test["p_tratado"] = modelo_tratado.predict_proba(X_test)[:, 1]
    test["p_control"] = modelo_control.predict_proba(X_test)[:, 1]
    test["uplift_estimado"] = test["p_tratado"] - test["p_control"]

    # 2. Modelo de riesgo/churn tradicional (para comparar en la Curva Qini)
    train_churn = train.copy()
    train_churn["churned"] = 1 - train_churn["responded_60d"]
    modelo_riesgo = LogisticRegression(max_iter=1000).fit(train_churn[FEATURES], train_churn["churned"])
    test["prob_churn"] = modelo_riesgo.predict_proba(X_test)[:, 1]

    return train, test


def valor_de_la_politica(test_df, seleccionados):
    sel = test_df.loc[seleccionados]
    valor_tratado = sel.loc[sel["treatment_group"] == "treatment", "utilidad_neta_60d"].mean()
    valor_control = sel.loc[sel["treatment_group"] == "control", "margin_60d"].mean()
    return valor_tratado - valor_control, len(sel)


def curva_qini(test_df, score_col, steps=40):
    orden = test_df.sort_values(score_col, ascending=False).reset_index(drop=True)
    es_tratado = (orden["treatment_group"] == "treatment").values
    respondio = orden["responded_60d"].values
    
    acumulado_tratados = np.cumsum(np.where(es_tratado, respondio, 0))
    acumulado_control = np.cumsum(np.where(~es_tratado, respondio, 0))
    n_tratados_acum = np.cumsum(es_tratado)
    n_control_acum = np.cumsum(~es_tratado)
    
    n = len(orden)
    xs, ys = [0.0], [0.0]
    for i in np.linspace(1, n, steps).astype(int):
        nt, nc = n_tratados_acum[i - 1], n_control_acum[i - 1]
        tasa_t = acumulado_tratados[i - 1] / nt if nt > 0 else 0
        tasa_c = acumulado_control[i - 1] / nc if nc > 0 else 0
        xs.append(i / n)
        ys.append((tasa_t - tasa_c) * n)
    return np.array(xs), np.array(ys)


def coeficiente_qini(xs, ys, n):
    area_modelo = np.trapezoid(ys, xs)
    area_azar = np.trapezoid(np.linspace(0, ys[-1], len(xs)), xs)
    return (area_modelo - area_azar) / n


# --- Título y carga de datos ---
st.title("☕ Laboratorio Final — Starbucks Rewards")
st.caption("Esta app permite optimizar el presupuesto de marketing identificando qué porcentaje de clientes contactar para maximizar las ganancias netas de la campaña.")

df = load_data()

# Control interactivo en la barra lateral
with st.sidebar:
    st.header("Parámetros de Campaña")
    pct_contactar = st.slider(
        "Porcentaje de la base a contactar (%):",
        min_value=10,
        max_value=100,
        value=40,
        step=10
    )
    st.info("Ajusta el control deslizante para calcular la rentabilidad y el volumen de clientes en tiempo real.")

train, test = entrenar_t_learner(df, test_size=0.30, seed=42)

# --- Métricas principales según el % elegido en el slider ---
n_c = int(len(test) * pct_contactar / 100)
idx_sel = test.sort_values("uplift_estimado", ascending=False).head(n_c).index
valor_por_cliente, n_sel = valor_de_la_politica(test, idx_sel)
valor_total = valor_por_cliente * n_sel

col1, col2, col3 = st.columns(3)
col1.metric("Clientes contactados", f"{n_sel:,} ({pct_contactar}%)")
col2.metric("Valor neto incremental por cliente", f"${valor_por_cliente:+.2f}")
col3.metric("Valor neto total acumulado", f"${valor_total:+,.2f}")

st.divider()

col_graf, col_tab = st.columns([1.2, 1])

# --- Curva Qini ---
with col_graf:
    st.subheader("1. Curva Qini del T-Learner")
    
    xs_uplift, ys_uplift = curva_qini(test, "uplift_estimado")
    xs_riesgo, ys_riesgo = curva_qini(test, "prob_churn")
    qini_uplift = coeficiente_qini(xs_uplift, ys_uplift, len(test))
    qini_riesgo = coeficiente_qini(xs_riesgo, ys_riesgo, len(test))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs_uplift, ys_uplift, color=GREEN, linewidth=2.5, label=f"T-learner (Qini={qini_uplift:+.3f})")
    ax.plot(xs_riesgo, ys_riesgo, color=ORANGE, linewidth=2.5, label=f"Riesgo/churn (Qini={qini_riesgo:+.3f})")
    ax.plot([0, 1], [0, ys_uplift[-1]], color=GRAY, linestyle="--", label="Al azar")
    ax.axvline(x=pct_contactar / 100, color="red", linestyle=":", label=f"Corte ({pct_contactar}%)")
    
    ax.set_xlabel("% de la base contactada (prueba)")
    ax.set_ylabel("Respuestas incrementales acumuladas")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

# --- Tabla de políticas del Paso 3 ---
with col_tab:
    st.subheader("2. Tabla de Políticas de Contacto")
    
    resultados = []
    for pct in range(10, 101, 10):
        nc_i = int(len(test) * pct / 100)
        idx_i = test.sort_values("uplift_estimado", ascending=False).head(nc_i).index
        v_cli, n_i = valor_de_la_politica(test, idx_i)
        resultados.append({
            "% Contactado": f"{pct}%",
            "Clientes": n_i,
            "Valor/Cliente ($)": round(v_cli, 2),
            "Valor Total ($)": round(v_cli * n_i, 2)
        })

    tabla_politicas = pd.DataFrame(resultados)
    st.dataframe(tabla_politicas, use_container_width=True, hide_index=True)
